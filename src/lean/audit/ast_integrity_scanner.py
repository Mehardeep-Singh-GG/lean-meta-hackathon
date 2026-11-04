"""
AST Anti-Hallucination & Static Code Integrity Scanner (Audit A14).

Scans all Python files in src/lean/ and experiments/ via Python's `ast` module to detect:
1. Defensive Clamping of Theoretical Quantities (e.g., max(0, delta_h), np.maximum(0.0, gamma)).
2. Hardcoded Metric Returns inside evaluation functions (returning constant dict literals).
3. Mocked Evaluation Sleeps (time.sleep inside evaluation/benchmark functions).
4. Calibration Data Leakage (any reference to vault/locked/ inside calibration/quantization modules).
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

THEORETICAL_IDENTIFIERS: Set[str] = {
    "delta_h",
    "half_life",
    "half_life_gap",
    "asymmetry_ratio",
    "half_life_ratio",
    "gamma",
    "alpha_emp",
    "knapsack_gap",
    "loocv_r2",
    "cliff_delta",
    "cliffs_delta",
    "kappa",
    "c_alg",
    "survival_auc",
}

EVAL_FUNCTION_PATTERNS = re.compile(
    r"^(evaluate_|run_live_agentic_episode|score_benchmark|compute_empirical_metrics)"
)

METRIC_DICT_KEYS: Set[str] = {
    "accuracy",
    "skill_acc",
    "knowledge_acc",
    "perplexity",
    "ppl",
    "auc",
    "survival_auc",
    "pass_rate",
    "exact_match",
    "f1_score",
}

CALIBRATION_MODULES: Set[str] = {
    "calibration_optimizer.py",
    "priced_quanta_v2.py",
    "knapsack.py",
    "hessian_sketch.py",
}

LOCKED_VAULT_PATTERNS = re.compile(
    r"(vault/locked|LEAN_UNLOCK|verify_unlock_token|open_locked_tranche)", re.IGNORECASE
)


def _extract_names_from_node(node: ast.AST) -> Set[str]:
    """Recursively collect all variable/attribute names referenced inside an AST subtree."""
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            names.add(child.attr.lower())
    return names


def _is_zero_constant(node: ast.AST) -> bool:
    """Return True if node is numeric literal 0 or 0.0."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) == 0.0
    return False


class LeanASTIntegrityVisitor(ast.NodeVisitor):
    """AST NodeVisitor enforcing Project LEAN A14 Scientific Integrity Rules."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        try:
            self.rel_path = str(filepath.resolve().relative_to(WORKSPACE_ROOT))
        except ValueError:
            self.rel_path = str(filepath)
        self.filename = filepath.name
        self.violations: List[Dict[str, Any]] = []
        self._current_function: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev_fn = self._current_function
        self._current_function = node.name

        if EVAL_FUNCTION_PATTERNS.match(node.name):
            self._check_eval_function_body(node)

        self.generic_visit(node)
        self._current_function = prev_fn

    def _check_eval_function_body(self, func_node: ast.FunctionDef) -> None:
        """Check evaluation functions for hardcoded constant metric dict returns or sleep mocking."""
        for child in ast.walk(func_node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                dict_node = child.value
                metric_keys_found = 0
                constant_metric_vals = 0
                for k_node, v_node in zip(dict_node.keys, dict_node.values):
                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                        k_str = k_node.value.lower()
                        if k_str in METRIC_DICT_KEYS:
                            metric_keys_found += 1
                            if isinstance(v_node, ast.Constant) and isinstance(v_node.value, (int, float)):
                                constant_metric_vals += 1
                if metric_keys_found > 0 and constant_metric_vals == metric_keys_found:
                    self.violations.append({
                        "rule": "A14_HARDCODED_EVAL_RETURN",
                        "severity": "S0",
                        "file": self.rel_path,
                        "line": child.lineno,
                        "function": func_node.name,
                        "detail": (
                            f"Evaluation function '{func_node.name}' returns a dictionary literal "
                            f"with {constant_metric_vals} hardcoded constant metric values."
                        ),
                    })

            if isinstance(child, ast.Call):
                func_names = _extract_names_from_node(child.func)
                if "sleep" in func_names:
                    self.violations.append({
                        "rule": "A14_MOCKED_SLEEP_IN_EVAL",
                        "severity": "S0",
                        "file": self.rel_path,
                        "line": child.lineno,
                        "function": func_node.name,
                        "detail": f"Evaluation function '{func_node.name}' invokes sleep() to mock execution time.",
                    })
                if "normal" in func_names and "rng" in func_names:
                    self.violations.append({
                        "rule": "A14_SYNTHETIC_JITTER_IN_EVAL",
                        "severity": "S0",
                        "file": self.rel_path,
                        "line": child.lineno,
                        "function": func_node.name,
                        "detail": f"Evaluation function '{func_node.name}' invokes rng.normal() to perturb evaluation scores.",
                    })
            if isinstance(child, ast.Name) and child.id == "kv_phase_jitter":
                self.violations.append({
                    "rule": "A14_SYNTHETIC_JITTER_IN_EVAL",
                    "severity": "S0",
                    "file": self.rel_path,
                    "line": child.lineno,
                    "function": func_node.name,
                    "detail": f"Evaluation function '{func_node.name}' references banned synthetic jitter 'kv_phase_jitter'.",
                })

    def visit_Call(self, node: ast.Call) -> None:
        """Detect defensive clamping on theoretical quantities (e.g. max(0, delta_h), clamp(min=0))."""
        func_names = _extract_names_from_node(node.func)
        clamp_funcs = {"max", "maximum", "clip", "clamp"}
        matched_clamp = func_names.intersection(clamp_funcs)

        if matched_clamp:
            has_zero_bound = any(_is_zero_constant(arg) for arg in node.args)
            for kw in node.keywords:
                if kw.arg in {"min", "a_min"} and _is_zero_constant(kw.value):
                    has_zero_bound = True

            if has_zero_bound:
                referenced_vars: Set[str] = set()
                for arg in node.args:
                    referenced_vars.update(_extract_names_from_node(arg))
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.AST):
                    referenced_vars.update(_extract_names_from_node(node.func.value))

                clamped_theory_vars = referenced_vars.intersection(THEORETICAL_IDENTIFIERS)
                if clamped_theory_vars:
                    self.violations.append({
                        "rule": "A14_DEFENSIVE_THEORY_CLAMPING",
                        "severity": "S0",
                        "file": self.rel_path,
                        "line": node.lineno,
                        "function": self._current_function,
                        "detail": (
                            f"Prohibited defensive clamping ({sorted(matched_clamp)[0]} with 0 bound) "
                            f"on theoretical quantity {sorted(clamped_theory_vars)}."
                        ),
                    })

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Detect any reference to vault/locked/ inside calibration or quantization modules."""
        if self.filename in CALIBRATION_MODULES and isinstance(node.value, str):
            if LOCKED_VAULT_PATTERNS.search(node.value):
                self.violations.append({
                    "rule": "A14_CALIBRATION_VAULT_LEAK",
                    "severity": "S0",
                    "file": self.rel_path,
                    "line": node.lineno,
                    "function": self._current_function,
                    "detail": (
                        f"Calibration/quantization module '{self.filename}' contains forbidden "
                        f"locked-vault string reference: '{node.value}'."
                    ),
                })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Detect forbidden imports of vault unlock routines inside calibration modules."""
        if self.filename in CALIBRATION_MODULES and node.module and "vault" in node.module:
            imported_names = {alias.name for alias in node.names}
            forbidden = {"verify_unlock_token", "get_active_lock_hashes", "LOCKED_SPLITS"}
            overlap = imported_names.intersection(forbidden)
            if overlap:
                self.violations.append({
                    "rule": "A14_CALIBRATION_VAULT_IMPORT_LEAK",
                    "severity": "S0",
                    "file": self.rel_path,
                    "line": node.lineno,
                    "function": self._current_function,
                    "detail": (
                        f"Calibration/quantization module '{self.filename}' imports forbidden "
                        f"vault unlock symbol(s): {sorted(overlap)}."
                    ),
                })
        self.generic_visit(node)


def scan_repository_ast(
    target_dirs: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Scan all Python files in src/lean/ and experiments/ using LeanASTIntegrityVisitor.
    Returns structured summary for Audit A14.
    """
    if target_dirs is None:
        target_dirs = ["src/lean", "experiments"]

    self_path = Path(__file__).resolve()
    files_scanned = 0
    all_violations: List[Dict[str, Any]] = []
    syntax_errors: List[Dict[str, Any]] = []

    for rel_dir in target_dirs:
        abs_dir = WORKSPACE_ROOT / rel_dir
        if not abs_dir.exists():
            continue
        for py_file in sorted(abs_dir.rglob("*.py")):
            if py_file.resolve() == self_path or "__pycache__" in py_file.parts:
                continue
            files_scanned += 1
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
                visitor = LeanASTIntegrityVisitor(py_file)
                visitor.visit(tree)
                all_violations.extend(visitor.violations)
            except SyntaxError as exc:
                syntax_errors.append({
                    "file": str(py_file.relative_to(WORKSPACE_ROOT)),
                    "line": exc.lineno,
                    "detail": f"SyntaxError during AST parse: {exc.msg}",
                })

    passed = (len(all_violations) == 0) and (len(syntax_errors) == 0) and (files_scanned > 0)
    return {
        "passed": passed,
        "files_scanned": files_scanned,
        "violation_count": len(all_violations),
        "syntax_error_count": len(syntax_errors),
        "violations": all_violations,
        "syntax_errors": syntax_errors,
    }


class ASTIntegrityScanner:
    """Compatibility wrapper class for ASTIntegrityScanner().scan_all()."""

    def __init__(self, target_dirs: List[str] | None = None) -> None:
        self.target_dirs = target_dirs

    def scan_all(self) -> Dict[str, Any]:
        return scan_repository_ast(self.target_dirs)


if __name__ == "__main__":
    report = scan_repository_ast()
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)

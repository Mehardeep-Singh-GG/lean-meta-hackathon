"""
Real, Deterministic Local Tool Sandbox for Milestone M17 Agentic Evaluation.

Implements 4 zero-mock tools with cryptographic SHA-256 input/output provenance:
1. PythonASTMathSandbox (`python_math_exec`): Safe AST-validated mathematical evaluator.
2. BM25SearchTool (`bm25_knowledge_lookup`): In-memory BM25 search over canonical & multi-hop corpus.
3. InMemorySQLiteTool (`sql_table_query`): Real SQLite3 in-memory relational engine with deterministic tables.
4. PytestCodeVerifier (`code_linter_check`): Real subprocess Python unit-test & traceback runner.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from src.lean.core.retrieval_store import BM25Index, build_canonical_knowledge_corpus


def _sha256_str(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PythonASTMathSandbox:
    """
    Executes mathematical expressions via strict Python AST parsing.
    Forbids arbitrary code execution while supporting full arithmetic, powers,
    and math module functions (sqrt, log, log2, exp, sin, cos, ceil, floor, factorial, gcd).
    """

    ALLOWED_BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    ALLOWED_UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }
    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"Unknown identifier '{node.id}'. Use numbers or math functions.")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_BINOPS:
                raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if op_type is ast.Pow and abs(float(right)) > 1000:
                raise ValueError("Exponent magnitude exceeds safety limit (1000).")
            return self.ALLOWED_BINOPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_UNARYOPS:
                raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
            return self.ALLOWED_UNARYOPS[op_type](self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "math":
                    func_name = node.func.attr
            if func_name not in self.ALLOWED_FUNCS:
                raise ValueError(f"Disallowed function call: '{func_name}'")
            eval_args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCS[func_name](*eval_args)
        raise ValueError(f"Unsupported AST syntax node: {type(node).__name__}")

    def execute(self, expression: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        expr_clean = str(expression).strip()
        try:
            parsed = ast.parse(expr_clean, mode="eval")
            val = self._eval_node(parsed)
            if isinstance(val, float) and val.is_integer():
                result_str = str(int(val))
            elif isinstance(val, float):
                result_str = f"{val:.6f}".rstrip("0").rstrip(".")
            else:
                result_str = str(val)
            status = "ok"
            output = result_str
        except Exception as exc:
            status = "error"
            output = f"MathSandboxError: {type(exc).__name__}: {str(exc)}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tool": "python_math_exec",
            "status": status,
            "input": expr_clean,
            "output": output,
            "latency_ms": round(latency_ms, 4),
            "io_sha256": _sha256_str(f"python_math_exec|{expr_clean}|{output}"),
        }


class BM25SearchTool:
    """
    Deterministic BM25 search tool over the canonical entity-relation corpus
    plus multi-hop scientific/engineering facts for ReAct chains.
    """

    EXTRA_AGENTIC_DOCS = [
        {
            "doc_id": 10001,
            "relation": "system_spec",
            "subject": "Project Titan",
            "target": "384",
            "text": "Project Titan operates a cluster of 384 liquid-cooled compute nodes in Zurich, where each node consumes 1.25 kilowatts at peak load.",
            "frequency": 5000,
        },
        {
            "doc_id": 10002,
            "relation": "system_spec",
            "subject": "Zurich Tariff",
            "target": "0.18",
            "text": "Industrial data centers in Zurich pay an electricity tariff of 0.18 CHF per kilowatt-hour (kWh) and have a carbon intensity of 24 grams CO2 per kWh.",
            "frequency": 4200,
        },
        {
            "doc_id": 10003,
            "relation": "system_spec",
            "subject": "Apollo Satellite",
            "target": "7200",
            "text": "The Apollo-9 telemetry satellite orbits at an altitude where its orbital period is 7200 seconds and transmits 45 megabits per second during a 12-minute ground pass.",
            "frequency": 3900,
        },
        {
            "doc_id": 10004,
            "relation": "system_spec",
            "subject": "CryoPump-X",
            "target": "42",
            "text": "CryoPump-X requires a maintenance overhaul every 420 operating hours. Each overhaul consumes 14 liters of liquid helium priced at 35 USD per liter.",
            "frequency": 3500,
        },
        {
            "doc_id": 10005,
            "relation": "system_spec",
            "subject": "Helios Battery",
            "target": "850",
            "text": "The Helios solid-state battery pack stores 850 watt-hours per kilogram and weighs 64 kilograms per module, operating at 92 percent round-trip efficiency.",
            "frequency": 4100,
        },
    ]

    def __init__(self, index: Optional[BM25Index] = None):
        if index is not None:
            self.index = index
        else:
            self.index = BM25Index()
            base_docs = build_canonical_knowledge_corpus()
            self.index.fit(base_docs + self.EXTRA_AGENTIC_DOCS)

    def execute(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_clean = str(query).strip()
        if not q_clean:
            status = "error"
            output = "BM25SearchError: Query string cannot be empty."
        else:
            hits = self.index.search(q_clean, top_k=max(1, min(5, int(top_k))))
            passages = [f"[{i+1}] (score={score:.2f}) {doc['text']}" for i, (doc, score) in enumerate(hits)]
            status = "ok"
            output = " | ".join(passages) if passages else "No matching passages found."
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tool": "bm25_knowledge_lookup",
            "status": status,
            "input": q_clean,
            "output": output,
            "latency_ms": round(latency_ms, 4),
            "io_sha256": _sha256_str(f"bm25_knowledge_lookup|{q_clean}|{output}"),
        }


class InMemorySQLiteTool:
    """
    Real in-memory SQLite relational database with deterministic engineering tables
    for testing multi-step SQL schema inspection, filtering, aggregation, and error recovery.
    """

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._seed_tables()

    def _seed_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE server_fleet (
                server_id TEXT PRIMARY KEY,
                datacenter TEXT,
                gpu_count INTEGER,
                memory_gb INTEGER,
                hourly_cost_usd REAL,
                status TEXT
            )
            """
        )
        fleet_rows = [
            ("srv-alpha-01", "zurich", 8, 640, 24.50, "active"),
            ("srv-alpha-02", "zurich", 8, 640, 24.50, "active"),
            ("srv-beta-01", "frankfurt", 4, 320, 11.20, "active"),
            ("srv-beta-02", "frankfurt", 4, 320, 11.20, "maintenance"),
            ("srv-gamma-01", "tokyo", 16, 1280, 52.00, "active"),
            ("srv-delta-01", "oregon", 8, 640, 21.00, "active"),
        ]
        cur.executemany("INSERT INTO server_fleet VALUES (?, ?, ?, ?, ?, ?)", fleet_rows)

        cur.execute(
            """
            CREATE TABLE department_budgets (
                dept_name TEXT PRIMARY KEY,
                primary_datacenter TEXT,
                quarterly_budget_usd REAL,
                headcount INTEGER
            )
            """
        )
        dept_rows = [
            ("compression_research", "zurich", 180000.0, 12),
            ("inference_systems", "frankfurt", 95000.0, 8),
            ("multimodal_pretraining", "tokyo", 420000.0, 20),
            ("edge_deployment", "oregon", 110000.0, 9),
        ]
        cur.executemany("INSERT INTO department_budgets VALUES (?, ?, ?, ?)", dept_rows)
        self.conn.commit()

    def execute(self, sql_query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_clean = str(sql_query).strip()
        if not q_clean.upper().startswith(("SELECT", "PRAGMA")):
            status = "error"
            output = "SQLSandboxError: Only read-only SELECT or PRAGMA queries are permitted."
        else:
            try:
                cur = self.conn.cursor()
                cur.execute(q_clean)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                formatted = [dict(zip(cols, r)) for r in rows]
                status = "ok"
                output = json.dumps(formatted)
            except Exception as exc:
                status = "error"
                output = (
                    f"SQLiteOperationalError: {str(exc)}. "
                    "Available tables: server_fleet(server_id, datacenter, gpu_count, memory_gb, hourly_cost_usd, status), "
                    "department_budgets(dept_name, primary_datacenter, quarterly_budget_usd, headcount)."
                )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tool": "sql_table_query",
            "status": status,
            "input": q_clean,
            "output": output,
            "latency_ms": round(latency_ms, 4),
            "io_sha256": _sha256_str(f"sql_table_query|{q_clean}|{output}"),
        }


class PytestCodeVerifier:
    """
    Executes generated Python code + unit test assertions in an isolated subprocess
    with a strict timeout, returning exact pass/fail status and real Python tracebacks.
    """

    def execute(
        self,
        code_str: str,
        test_assertions: Optional[List[str]] = None,
        timeout_s: float = 4.0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        assertions_block = "\n".join(test_assertions or [])
        full_script = f"{code_str}\n\n# --- Automated Unit Test Verification ---\n{assertions_block}\nprint('ALL_TESTS_PASSED_OK')\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", full_script],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if proc.returncode == 0 and "ALL_TESTS_PASSED_OK" in proc.stdout:
                status = "ok"
                output = "ALL_TESTS_PASSED_OK"
            else:
                status = "error"
                err_msg = (proc.stderr or proc.stdout).strip()
                tb_lines = err_msg.splitlines()[-6:]
                output = "PythonTestFailure:\n" + "\n".join(tb_lines)
        except subprocess.TimeoutExpired:
            status = "error"
            output = f"PythonTimeoutError: Execution exceeded {timeout_s:.1f}s timeout (possible infinite loop)."
        except Exception as exc:
            status = "error"
            output = f"VerifierExecutionError: {type(exc).__name__}: {str(exc)}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "tool": "code_linter_check",
            "status": status,
            "input": code_str[:240],
            "output": output,
            "latency_ms": round(latency_ms, 4),
            "io_sha256": _sha256_str(f"code_linter_check|{code_str}|{assertions_block}|{output}"),
        }


class AgentToolSuite:
    """Unified router dispatching validated JSON tool calls to the 4 deterministic local tools."""

    def __init__(self, bm25_index: Optional[BM25Index] = None):
        self.math_tool = PythonASTMathSandbox()
        self.bm25_tool = BM25SearchTool(index=bm25_index)
        self.sql_tool = InMemorySQLiteTool()
        self.code_tool = PytestCodeVerifier()

    def dispatch(self, tool_name: str, tool_arg: str) -> Dict[str, Any]:
        name = str(tool_name).strip().lower()
        if name == "python_math_exec":
            return self.math_tool.execute(tool_arg)
        if name == "bm25_knowledge_lookup":
            return self.bm25_tool.execute(tool_arg)
        if name == "sql_table_query":
            return self.sql_tool.execute(tool_arg)
        if name == "code_linter_check":
            return self.code_tool.execute(tool_arg)
        return {
            "tool": name,
            "status": "error",
            "input": tool_arg,
            "output": (
                f"UnknownToolError: Tool '{tool_name}' is not registered. "
                "Valid tools: ['python_math_exec', 'bm25_knowledge_lookup', 'sql_table_query', 'code_linter_check']."
            ),
            "latency_ms": 0.01,
            "io_sha256": _sha256_str(f"unknown|{name}|{tool_arg}"),
        }

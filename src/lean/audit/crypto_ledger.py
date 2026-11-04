"""
Immutable Cryptographic Hardware & Provenance Ledger for Project LEAN (Milestone M13).

File: src/lean/audit/crypto_ledger.py
Maintains audit/CRYPTO_LEDGER.jsonl using the Merkle / Hash-Chained Receipt Equation:
R_k = SHA256(R_{k-1} || GitCommitSHA || ScriptSHA256 || ConfigSHA256 ||
             HardwareFingerprintSHA256 || InputVaultSHA256 || OutputArtifactSHA256 || WallClockUTC)
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AUDIT_DIR = REPO_ROOT / "audit"
CRYPTO_LEDGER_PATH = AUDIT_DIR / "CRYPTO_LEDGER.jsonl"
GENESIS_PREV_HASH = "0" * 64


class CryptoLedgerTamperError(RuntimeError):
    """Raised when any entry in audit/CRYPTO_LEDGER.jsonl fails hash-chain verification."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_git_state_digest(repo_root: Path = REPO_ROOT) -> Tuple[str, str, bool]:
    """
    Compute H_git = SHA256(c_head || SHA256(git_status_and_diff)).
    Returns (H_git, c_head, is_dirty).
    """
    try:
        c_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uall"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        )
        diff_out = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        )
        dirty = bool(status_out.strip() or diff_out.strip())
        diff_sha = sha256_bytes((status_out + "\n" + diff_out).encode("utf-8"))
        h_git = sha256_bytes(f"{c_head}|{diff_sha}".encode("utf-8"))
        return h_git, c_head, dirty
    except Exception:
        fallback = sha256_bytes(b"NO_GIT_REPO")
        return fallback, "UNKNOWN_HEAD", True


def compute_script_and_src_digest(script_path: Path, repo_root: Path = REPO_ROOT) -> str:
    """
    Compute H_script = SHA256(SHA256(script) || MerkleRoot(src/lean/**/*.py)).
    """
    script_sha = sha256_file(script_path) if script_path.exists() and script_path.is_file() else sha256_bytes(b"MISSING")
    src_dir = repo_root / "src" / "lean"
    src_records: List[str] = []
    if src_dir.exists():
        for p in sorted(src_dir.rglob("*.py")):
            if p.is_file():
                rel = p.relative_to(repo_root).as_posix()
                src_records.append(f"{rel}:{sha256_file(p)}")
    src_merkle = sha256_bytes("\n".join(src_records).encode("utf-8"))
    return sha256_bytes(f"{script_sha}|{src_merkle}".encode("utf-8"))


def compute_file_list_digest(files: List[Path], repo_root: Path = REPO_ROOT) -> Tuple[str, Dict[str, str]]:
    """
    Compute deterministic digest over an ordered list of input or output files.
    Returns (digest_hex, {relpath: sha256}).
    """
    manifest: Dict[str, str] = {}
    lines: List[str] = []
    for p in sorted(files, key=lambda x: str(x)):
        resolved = p.resolve()
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            rel = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel = resolved.name
        f_sha = sha256_file(resolved)
        manifest[rel] = f_sha
        lines.append(f"{rel}:{f_sha}\n")
    digest = sha256_bytes("".join(lines).encode("utf-8"))
    return digest, manifest


def collect_local_hardware_fingerprint() -> Dict[str, Any]:
    """Collect hardware telemetry on local Mac (Apple Silicon MPS/sysctl) or Linux node."""
    hw: Dict[str, Any] = {
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": sys.version.split()[0],
        "host_mem_total_bytes": 0,
        "rss_peak_bytes": 0,
        "primary_gpu_uuid": "LOCAL-APPLE-SILICON-MPS" if platform.system() == "Darwin" else "LOCAL-CPU",
        "primary_gpu_name": platform.processor() or "Apple Silicon",
        "mps_available": False,
        "cuda_available": False,
    }
    if platform.system() == "Darwin":
        try:
            hw["host_mem_total_bytes"] = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            cpu_brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if cpu_brand:
                hw["processor"] = cpu_brand
                hw["primary_gpu_name"] = cpu_brand
        except Exception:
            pass
    elif Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                nums = re.findall(r"\d+", line)
                if nums:
                    hw["host_mem_total_bytes"] = int(nums[0]) * 1024

    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        hw["rss_peak_bytes"] = int(ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024)
    except Exception:
        pass

    try:
        import torch
        hw["torch_version"] = torch.__version__
        hw["cuda_available"] = bool(torch.cuda.is_available())
        hw["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        pass
    return hw


def compute_receipt_hash(
    prev_receipt_sha256: str,
    git_commit_sha256: str,
    script_sha256: str,
    config_sha256: str,
    hardware_fingerprint_sha256: str,
    input_vault_sha256: str,
    output_artifact_sha256: str,
    wall_clock_utc: str,
) -> str:
    """
    Evaluate the exact M13 Merkle / Hash-Chained Receipt Equation:
    R_k = SHA256(R_{k-1} || GitCommitSHA || ScriptSHA256 || ConfigSHA256 ||
                 HardwareFingerprintSHA256 || InputVaultSHA256 || OutputArtifactSHA256 || WallClockUTC)
    """
    preimage = "|".join([
        prev_receipt_sha256,
        git_commit_sha256,
        script_sha256,
        config_sha256,
        hardware_fingerprint_sha256,
        input_vault_sha256,
        output_artifact_sha256,
        wall_clock_utc,
    ])
    return sha256_bytes(preimage.encode("utf-8"))


def load_ledger_entries(ledger_path: Path = CRYPTO_LEDGER_PATH) -> List[Dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries


def verify_ledger_chain(ledger_path: Path = CRYPTO_LEDGER_PATH) -> Tuple[bool, int, str]:
    """
    Verify every receipt R_0 ... R_k in audit/CRYPTO_LEDGER.jsonl.
    Raises CryptoLedgerTamperError if any hash link or internal digest is invalid.
    """
    entries = load_ledger_entries(ledger_path)
    prev_hash = GENESIS_PREV_HASH
    for idx, entry in enumerate(entries):
        if entry.get("index") != idx:
            raise CryptoLedgerTamperError(f"Ledger index mismatch at row {idx}: got {entry.get('index')}")
        if entry.get("prev_receipt_sha256") != prev_hash:
            raise CryptoLedgerTamperError(
                f"Broken chain link at index {idx}: expected prev_receipt_sha256={prev_hash}, "
                f"got {entry.get('prev_receipt_sha256')}"
            )

        expected_cfg_sha = sha256_bytes(canonical_json_bytes(entry.get("config", {})))
        if entry.get("config_sha256") != expected_cfg_sha:
            raise CryptoLedgerTamperError(f"Config SHA-256 tamper detected at index {idx}")

        expected_hw_sha = sha256_bytes(canonical_json_bytes(entry.get("hardware_telemetry", {})))
        if entry.get("hardware_fingerprint_sha256") != expected_hw_sha:
            raise CryptoLedgerTamperError(f"Hardware fingerprint SHA-256 tamper detected at index {idx}")

        recomputed_r_k = compute_receipt_hash(
            prev_receipt_sha256=entry["prev_receipt_sha256"],
            git_commit_sha256=entry["git_commit_sha256"],
            script_sha256=entry["script_sha256"],
            config_sha256=entry["config_sha256"],
            hardware_fingerprint_sha256=entry["hardware_fingerprint_sha256"],
            input_vault_sha256=entry["input_vault_sha256"],
            output_artifact_sha256=entry["output_artifact_sha256"],
            wall_clock_utc=entry["wall_clock_utc"],
        )
        if entry.get("receipt_sha256") != recomputed_r_k:
            raise CryptoLedgerTamperError(
                f"Receipt signature mismatch at index {idx}: stored={entry.get('receipt_sha256')}, "
                f"recomputed={recomputed_r_k}"
            )
        prev_hash = recomputed_r_k

    return True, len(entries), prev_hash


def append_signed_receipt(
    milestone: str,
    job_id: str,
    execution_mode: str,
    script_path: Path,
    config: Dict[str, Any],
    hardware_telemetry: Optional[Dict[str, Any]] = None,
    input_files: Optional[List[Path]] = None,
    output_files: Optional[List[Path]] = None,
    repo_root: Path = REPO_ROOT,
    ledger_path: Path = CRYPTO_LEDGER_PATH,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create, sign, verify, and append receipt R_k to audit/CRYPTO_LEDGER.jsonl."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _, current_len, prev_hash = verify_ledger_chain(ledger_path)

    git_digest, git_head, git_dirty = compute_git_state_digest(repo_root)
    script_digest = compute_script_and_src_digest(script_path, repo_root)
    cfg_digest = sha256_bytes(canonical_json_bytes(config))

    hw_payload = hardware_telemetry if hardware_telemetry else collect_local_hardware_fingerprint()
    hw_digest = sha256_bytes(canonical_json_bytes(hw_payload))

    in_digest, in_manifest = compute_file_list_digest(input_files or [], repo_root)
    out_digest, out_manifest = compute_file_list_digest(output_files or [], repo_root)
    wall_clock_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    r_k = compute_receipt_hash(
        prev_receipt_sha256=prev_hash,
        git_commit_sha256=git_digest,
        script_sha256=script_digest,
        config_sha256=cfg_digest,
        hardware_fingerprint_sha256=hw_digest,
        input_vault_sha256=in_digest,
        output_artifact_sha256=out_digest,
        wall_clock_utc=wall_clock_utc,
    )

    entry = {
        "index": current_len,
        "milestone": milestone,
        "job_id": job_id,
        "execution_mode": execution_mode,
        "wall_clock_utc": wall_clock_utc,
        "prev_receipt_sha256": prev_hash,
        "receipt_sha256": r_k,
        "git_head_commit": git_head,
        "git_working_tree_dirty": git_dirty,
        "git_commit_sha256": git_digest,
        "script_relpath": script_path.name if not script_path.is_absolute() else (
            script_path.relative_to(repo_root).as_posix() if str(script_path).startswith(str(repo_root)) else str(script_path)
        ),
        "script_sha256": script_digest,
        "config": config,
        "config_sha256": cfg_digest,
        "hardware_telemetry": hw_payload,
        "hardware_fingerprint_sha256": hw_digest,
        "input_manifest": in_manifest,
        "input_vault_sha256": in_digest,
        "output_manifest": out_manifest,
        "output_artifact_sha256": out_digest,
        "extra_metadata": extra_metadata or {},
    }

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

    return entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Project LEAN M13 — Cryptographic Provenance Ledger CLI")
    parser.add_argument("--verify-all", action="store_true", help="Verify full Merkle hash chain in CRYPTO_LEDGER.jsonl")
    parser.add_argument("--ledger-path", type=str, default=str(CRYPTO_LEDGER_PATH), help="Path to ledger JSONL")
    args = parser.parse_args(argv)

    path = Path(args.ledger_path)
    if args.verify_all:
        ok, count, head_hash = verify_ledger_chain(path)
        print(json.dumps({
            "ok": ok,
            "verified_entries": count,
            "head_receipt_sha256": head_hash,
            "ledger_path": str(path),
        }, indent=2))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

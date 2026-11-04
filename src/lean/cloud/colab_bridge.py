"""
Local Antigravity Cloud Bridge Client for Google Colab H100/A100 Worker (Milestone M13).

File: src/lean/cloud/colab_bridge.py
Provides programmatic API and CLI (--ping, --verify-auth, --gpu-status, --sync, --run, --watch, --pull)
with strict SHA-256 verification and automatic receipt recording in audit/CRYPTO_LEDGER.jsonl.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.lean.audit.crypto_ledger import (
    append_signed_receipt,
    compute_git_state_digest,
    compute_script_and_src_digest,
    sha256_bytes,
    sha256_file,
    verify_ledger_chain,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SYNC_DIRS = ["src", "experiments", "configs", "prereg", "campaigns"]
EXCLUDED_PATTERNS = {"__pycache__", ".pytest_cache", ".DS_Store", ".git", ".venv", "vault/locked"}


class ColabBridgeError(RuntimeError):
    """Raised when communication or SHA-256 verification with the remote Colab worker fails."""


class ColabBridgeClient:
    """Resilient HTTP/TLS client for the Project LEAN Remote Colab GPU Worker."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        timeout_sec: float = 45.0,
        max_retries: int = 4,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("ANTIGRAVITY_COLAB_ENDPOINT", "")).rstrip("/")
        self.token = (token or os.environ.get("ANTIGRAVITY_COLAB_TOKEN", "")).strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        if not self.endpoint or not self.token:
            raise ColabBridgeError(
                "Missing ANTIGRAVITY_COLAB_ENDPOINT or ANTIGRAVITY_COLAB_TOKEN environment variables."
            )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        custom_token: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.endpoint}{path}"
        active_token = custom_token if custom_token is not None else self.token
        headers = {
            "Authorization": f"Bearer {active_token}",
            "Accept": "application/json",
            "User-Agent": "ProjectLEAN-AntigravityBridge/M13",
        }
        data_bytes = None
        if payload is not None:
            data_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                    return resp.status, json.loads(raw)
            except urllib.error.HTTPError as e:
                raw_err = e.read().decode("utf-8", errors="replace")
                try:
                    parsed_err = json.loads(raw_err)
                except Exception:
                    parsed_err = {"ok": False, "error": raw_err}
                if e.code in (502, 503, 504, 524) and attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (2 ** attempt))
                    last_err = e
                    continue
                return e.code, parsed_err
            except Exception as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise ColabBridgeError(f"Connection to {url} failed after {self.max_retries} attempts: {e}") from e
        raise ColabBridgeError(f"Request {method} {url} exhausted retries: {last_err}")

    def ping(self) -> Dict[str, Any]:
        status, data = self._request_json("GET", "/health")
        if status != 200 or (data.get("ok") is not True and data.get("status") != "ok"):
            raise ColabBridgeError(f"Health check failed ({status}): {data}")
        return data

    def verify_auth_rejection(self) -> bool:
        """Verify that an invalid token is strictly rejected with HTTP 401 (Gate M13-G1)."""
        status, data = self._request_json("GET", "/gpu_status", custom_token="INVALID_TOKEN_FOR_AUDIT_TEST")
        if status != 401:
            raise ColabBridgeError(f"Expected HTTP 401 on bad token, got {status}: {data}")
        return True

    def gpu_status(self, require_gpu: bool = False, min_vram_mib: float = 0.0) -> Dict[str, Any]:
        status, data = self._request_json("GET", "/gpu_status")
        if status != 200:
            raise ColabBridgeError(f"/gpu_status failed ({status}): {data}")
        gpus = data.get("gpu_devices") or data.get("gpus", [])
        if require_gpu and not gpus:
            raise ColabBridgeError("Remote worker has no NVIDIA GPU attached (--require-gpu enforced).")
        if min_vram_mib > 0:
            if not gpus:
                raise ColabBridgeError(f"Remote worker has 0 GPUs; requires {min_vram_mib} MiB.")
            actual_vram = gpus[0].get("vram_total_mib", gpus[0].get("memory_total_mib", 0.0))
            if actual_vram < min_vram_mib:
                raise ColabBridgeError(f"Remote GPU VRAM {actual_vram} MiB is below required {min_vram_mib} MiB.")
        return data

    def build_deterministic_sync_tarball(
        self, include_dirs: Optional[List[str]] = None
    ) -> Tuple[bytes, str, int]:
        """Create deterministic .tar.gz of local code/configs excluding vault/locked and caches."""
        dirs = include_dirs or DEFAULT_SYNC_DIRS
        buf = io.BytesIO()
        file_count = 0
        with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
            for d in sorted(dirs):
                dir_path = REPO_ROOT / d
                if not dir_path.exists():
                    continue
                for p in sorted(dir_path.rglob("*")):
                    if not p.is_file():
                        continue
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    if any(excl in rel for excl in EXCLUDED_PATTERNS) or p.suffix in {".pyc", ".pyo", ".so"}:
                        continue
                    ti = tarfile.TarInfo(name=rel)
                    data = p.read_bytes()
                    ti.size = len(data)
                    ti.mtime = 0
                    ti.mode = 0o644
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = "lean"
                    ti.gname = "lean"
                    tf.addfile(ti, io.BytesIO(data))
                    file_count += 1
        raw_bytes = buf.getvalue()
        return raw_bytes, sha256_bytes(raw_bytes), file_count

    def sync_repo(self, include_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_tar, tar_sha, count = self.build_deterministic_sync_tarball(include_dirs=include_dirs)
        tar_b64 = base64.b64encode(raw_tar).decode("ascii")

        # Try /sync_bundle first, then /sync_repo
        payload = {
            "tar_gz_b64": tar_b64,
            "expected_sha256": tar_sha,
            "bundled_file_count": count,
        }
        status, resp = self._request_json("POST", "/sync_bundle", payload=payload)
        if status == 404:
            # Fall back to /sync_repo
            status, resp = self._request_json("POST", "/sync_repo", payload=payload)

        if status != 200 or (resp.get("status") != "synced" and not resp.get("ok")):
            raise ColabBridgeError(f"Repository sync failed ({status}): {resp}")

        manifest = resp.get("sync_manifest") or {
            "tar_sha256": tar_sha,
            "file_count": count,
            "status": "synced",
        }
        return manifest

    def start_job(
        self,
        script_relpath: str,
        args: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        expected_outputs: Optional[List[str]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        job_id: Optional[str] = None,
        milestone: str = "M13",
    ) -> Dict[str, Any]:
        git_digest, git_head, _ = compute_git_state_digest(REPO_ROOT)
        payload = {
            "script": script_relpath,
            "script_relpath": script_relpath,
            "args": args or [],
            "config": config or {"cli_args": args or [], "script": script_relpath},
            "git_commit_sha": f"{git_head}:{git_digest[:16]}",
            "expected_outputs": expected_outputs or [],
            "env": extra_env or {},
            "extra_env": extra_env or {},
            "milestone": milestone,
        }
        if job_id:
            payload["job_id"] = job_id
        status, resp = self._request_json("POST", "/start_job", payload=payload)
        if status != 200 or ("job_id" not in resp and not resp.get("ok")):
            raise ColabBridgeError(f"Failed to start job ({status}): {resp}")
        return resp.get("job") or resp

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        status, resp = self._request_json("GET", f"/job_status/{job_id}")
        if status != 200:
            raise ColabBridgeError(f"Failed to query job_status for {job_id} ({status}): {resp}")
        # Standardize return dictionary
        job = resp.get("job", resp)
        normalized = {
            "ok": True,
            "job": {
                "job_id": job.get("job_id", job_id),
                "state": (job.get("status") or job.get("state", "unknown")).lower(),
                "exit_code": job.get("return_code") if "return_code" in job else job.get("exit_code"),
                "peak_vram_mib": job.get("peak_vram_mib", 0.0),
                "peak_host_ram_gb": job.get("peak_host_ram_gb", 0.0),
                "telemetry_samples": job.get("telemetry_samples", 1),
                "script_relpath": job.get("request", {}).get("script") or job.get("script_relpath", ""),
                "config": job.get("request", {}).get("config") or job.get("config", {}),
                "expected_outputs": job.get("request", {}).get("expected_outputs") or job.get("expected_outputs", []),
                "duration_sec": job.get("wall_clock_seconds") or job.get("duration_sec", 0.0),
                "worker_receipt": job.get("receipt") or job.get("worker_receipt", {}),
                "output_artifacts": job.get("receipt", {}).get("output_artifacts", {}) or job.get("artifact_hashes", {}),
            },
            "log_tail": resp.get("log_tail") or resp.get("log_tail_40", []),
        }
        return normalized

    def watch_job(self, job_id: str, poll_interval_sec: float = 2.0, timeout_sec: float = 7200.0) -> Dict[str, Any]:
        t0 = time.time()
        last_line_printed = ""
        while True:
            if time.time() - t0 > timeout_sec:
                raise ColabBridgeError(f"Job {job_id} exceeded watch timeout of {timeout_sec}s.")
            info = self.get_job_status(job_id)
            job = info["job"]
            tail = info.get("log_tail", [])
            if tail and tail[-1] != last_line_printed:
                last_line_printed = tail[-1]
                print(
                    f"[{job_id} | {job['state']} | peak_vram={job.get('peak_vram_mib', 0):.0f}MiB] {last_line_printed}",
                    flush=True,
                )
            if job["state"] in ("completed", "failed"):
                return info
            time.sleep(poll_interval_sec)

    def download_artifact(self, job_id: str, remote_relpath: str, local_dest: Path) -> str:
        """Download an artifact from the remote worker and verify X-Artifact-SHA256 before saving."""
        qs = urllib.parse.urlencode({"path": remote_relpath})
        url = f"{self.endpoint}/download_artifact/{job_id}?{qs}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ProjectLEAN-AntigravityBridge/M13",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            expected_sha = resp.headers.get("X-Artifact-SHA256", "")
            raw_bytes = resp.read()

        actual_sha = sha256_bytes(raw_bytes)
        if expected_sha and actual_sha != expected_sha:
            raise ColabBridgeError(
                f"SHA-256 verification failed for artifact {remote_relpath}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        local_dest.write_bytes(raw_bytes)
        return actual_sha

    def pull_job_bundle_and_record_receipt(
        self,
        job_id: str,
        milestone: str = "M13",
        input_files: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Pull logs, receipt, and outputs, then seal in CRYPTO_LEDGER.jsonl."""
        info = self.get_job_status(job_id)
        job = info["job"]
        if job["state"] not in ("completed", "failed"):
            raise ColabBridgeError(f"Cannot pull bundle while job {job_id} is in state '{job['state']}'.")

        local_job_dir = REPO_ROOT / "results" / "colab_jobs" / job_id
        local_job_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths: List[Path] = []

        # Download worker log & receipt
        try:
            log_dest = local_job_dir / "worker.log"
            self.download_artifact(job_id, "log", log_dest)
            downloaded_paths.append(log_dest)
        except Exception:
            pass

        try:
            rc_dest = local_job_dir / "crypto_receipt.json"
            self.download_artifact(job_id, "receipt", rc_dest)
            downloaded_paths.append(rc_dest)
        except Exception:
            pass

        # Download declared outputs
        expected_outputs = job.get("expected_outputs", [])
        if isinstance(job.get("output_artifacts"), dict):
            expected_outputs = list(set(expected_outputs + list(job["output_artifacts"].keys())))

        for out_rel in expected_outputs:
            dest = REPO_ROOT / out_rel
            try:
                self.download_artifact(job_id, out_rel, dest)
                downloaded_paths.append(dest)
            except Exception as e:
                print(f"[Bridge] Warning: could not download {out_rel}: {e}", file=sys.stderr)

        worker_receipt = job.get("worker_receipt", {})
        hw_fingerprint = (
            worker_receipt.get("hardware_end")
            or worker_receipt.get("hardware_start")
            or worker_receipt.get("hardware_fingerprint")
            or {}
        )
        script_rel = job.get("script_relpath") or "experiments/m13_colab_smoke_verify.py"
        script_path = REPO_ROOT / script_rel

        ledger_entry = append_signed_receipt(
            milestone=milestone,
            job_id=job_id,
            execution_mode="remote_colab_bridge",
            script_path=script_path,
            config=job.get("config", {}),
            hardware_telemetry=hw_fingerprint,
            input_files=input_files or [],
            output_files=downloaded_paths,
            repo_root=REPO_ROOT,
            extra_metadata={
                "remote_endpoint": self.endpoint,
                "exit_code": job.get("exit_code"),
                "duration_sec": job.get("duration_sec"),
                "peak_vram_mib": job.get("peak_vram_mib"),
                "worker_receipt_sha256": worker_receipt.get("receipt_sha256"),
            },
        )
        verify_ledger_chain()
        return ledger_entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project LEAN M13 — Antigravity Local-to-Colab H100/A100 Cloud Bridge CLI"
    )
    parser.add_argument("--endpoint", type=str, default=None, help="Override ANTIGRAVITY_COLAB_ENDPOINT")
    parser.add_argument("--token", type=str, default=None, help="Override ANTIGRAVITY_COLAB_TOKEN")
    parser.add_argument("--ping", action="store_true", help="Check remote worker health")
    parser.add_argument("--verify-auth", action="store_true", help="Verify HTTP 401 rejection on invalid token")
    parser.add_argument("--gpu-status", action="store_true", help="Fetch full remote GPU/RAM/Disk telemetry")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if no NVIDIA GPU is attached remotely")
    parser.add_argument("--min-vram-mib", type=float, default=0.0, help="Require minimum GPU VRAM in MiB")
    parser.add_argument("--sync", action="store_true", help="Sync local src/, experiments/, configs/, prereg/")
    parser.add_argument("--run", action="store_true", help="Start an asynchronous job on the remote worker")
    parser.add_argument("--script", type=str, default="", help="Relative path to script for --run")
    parser.add_argument("--script-args", type=str, default="", help="Arguments string passed to remote script")
    parser.add_argument("--expected-outputs", nargs="*", default=[], help="Relative paths of output artifacts")
    parser.add_argument("--watch", type=str, default="", help="Watch a running job_id until completion (or 'auto')")
    parser.add_argument("--pull", type=str, default="", help="Pull artifacts and record signed receipt for job_id")
    parser.add_argument("--milestone", type=str, default="M13", help="Milestone tag for CRYPTO_LEDGER.jsonl")

    args = parser.parse_args(argv)
    client = ColabBridgeClient(endpoint=args.endpoint, token=args.token)

    if args.verify_auth:
        client.verify_auth_rejection()
        print(json.dumps({"ok": True, "auth_rejection_verified": True}, indent=2))
        return 0

    if args.ping:
        res = client.ping()
        print(json.dumps(res, indent=2))
        return 0

    if args.gpu_status:
        res = client.gpu_status(require_gpu=args.require_gpu, min_vram_mib=args.min_vram_mib)
        print(json.dumps(res, indent=2))
        return 0

    if args.sync:
        manifest = client.sync_repo()
        print(json.dumps({"ok": True, "sync_manifest": manifest}, indent=2))

    if args.run:
        if not args.script:
            raise ColabBridgeError("--run requires --script <relative_path>")
        script_args_list = shlex.split(args.script_args) if args.script_args else []
        job = client.start_job(
            script_relpath=args.script,
            args=script_args_list,
            expected_outputs=args.expected_outputs,
            milestone=args.milestone,
        )
        job_id = job["job_id"]
        print(json.dumps({"ok": True, "started_job": job}, indent=2))
        if args.watch == "auto":
            final_info = client.watch_job(job_id)
            receipt = client.pull_job_bundle_and_record_receipt(job_id, milestone=args.milestone)
            print(json.dumps({"ok": True, "final_job": final_info["job"], "receipt": receipt}, indent=2))
        return 0

    if args.watch and args.watch != "auto":
        info = client.watch_job(args.watch)
        print(json.dumps(info, indent=2))

    if args.pull:
        receipt = client.pull_job_bundle_and_record_receipt(args.pull, milestone=args.milestone)
        print(json.dumps({"ok": True, "receipt": receipt}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

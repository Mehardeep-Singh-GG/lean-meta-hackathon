#!/usr/bin/env python3
# ==============================================================================
# ANTIGRAVITY <-> GOOGLE COLAB PRO+ REMOTE GPU WORKER FOR PROJECT LEAN (30B)
# ==============================================================================
# Instructions:
# 1. Open Google Colab Pro+ (https://colab.research.google.com)
# 2. Select Runtime -> Change runtime type -> H100 GPU or A100 GPU + High-RAM
# 3. Paste this entire file into the first cell and click Run.
# 4. Copy the printed ANTIGRAVITY_COLAB_ENDPOINT and ANTIGRAVITY_COLAB_TOKEN
#    back into the Antigravity chat!
# ==============================================================================

import os
import subprocess
import sys
import secrets
from pathlib import Path

print("[1/3] Installing Project LEAN 30B Dependencies (PyTorch, Transformers, Safetensors, Cloudflared)...")
subprocess.run(
    "pip install -q transformers accelerate safetensors datasets scipy scikit-learn psutil pyyaml huggingface_hub",
    shell=True,
    check=False,
)

WORKDIR = Path("/content/lean_30b_worker")
WORKDIR.mkdir(parents=True, exist_ok=True)

print("[2/3] Writing colab_lean_worker_daemon.py...")
(WORKDIR / "colab_lean_worker_daemon.py").write_text(r'''#!/usr/bin/env python3
"""
colab_lean_worker_daemon.py
===========================
Self-contained Zero-Dependency HTTP/REST GPU Worker Daemon for Project LEAN (30B -> 8GB RAM).
Runs on Google Colab Pro+ (H100 80GB / A100 80GB / L4) and exposes a Cloudflare Tunnel
endpoint with Bearer token authentication and SHA-256 cryptographic hardware receipts.

REST Protocol:
  GET  /health
  GET  /gpu_status
  POST /upload_file
  POST /sync_bundle
  POST /start_job
  GET  /job_status/<job_id>
  GET  /download_artifact/<job_id>?path=<relpath>
"""
import os
import re
import sys
import json
import time
import uuid
import tarfile
import io
import base64
import shutil
import socket
import platform
import hashlib
import urllib.parse
import argparse
import threading
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List

WORKDIR = os.path.abspath("./lean_colab_workspace")
REPO_DIR = os.path.join(WORKDIR, "repo")
JOBS_DIR = os.path.join(WORKDIR, "jobs")
LEDGER_PATH = os.path.join(WORKDIR, "colab_crypto_ledger.jsonl")
for _d in (WORKDIR, REPO_DIR, JOBS_DIR):
    os.makedirs(_d, exist_ok=True)

API_TOKEN = os.environ.get("ANTIGRAVITY_COLAB_TOKEN", "lean-30b-secret-token-2026")
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_hardware_credentials() -> Dict[str, Any]:
    """Captures verifiable GPU UUID, VRAM, Host RAM, CPU, and software versions."""
    info: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(shutil.disk_usage(WORKDIR).free / (1024**3), 3),
        "host_ram": {},
        "gpu_devices": [],
        "torch_info": {},
    }
    # Host RAM from /proc/meminfo or psutil
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["host_ram"] = {
            "total_gb": round(vm.total / (1024**3), 3),
            "available_gb": round(vm.available / (1024**3), 3),
            "used_gb": round(vm.used / (1024**3), 3),
            "percent": vm.percent,
        }
    except Exception:
        pass

    # NVIDIA GPU UUID & Telemetry via nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 11:
                info["gpu_devices"].append({
                    "index": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "name": parts[2],
                    "driver_version": parts[3],
                    "pci_bus_id": parts[4],
                    "vram_total_mib": float(parts[5]),
                    "vram_used_mib": float(parts[6]),
                    "vram_free_mib": float(parts[7]),
                    "utilization_gpu_pct": parts[8],
                    "temperature_c": parts[9],
                    "power_draw_w": parts[10],
                })
    except Exception as e:
        info["nvidia_smi_error"] = str(e)

    # PyTorch CUDA check
    try:
        import torch
        info["torch_info"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        info["torch_error"] = str(e)

    info["hardware_fingerprint_sha256"] = sha256_bytes(
        json.dumps(
            {
                "gpus": [(g.get("gpu_uuid"), g.get("name"), g.get("vram_total_mib")) for g in info["gpu_devices"]],
                "host_ram_total_gb": info["host_ram"].get("total_gb"),
                "torch": info["torch_info"],
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return info


def append_crypto_ledger(record: Dict[str, Any]) -> Dict[str, Any]:
    prev_hash = "0" * 64
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as lf:
            lines = [l.strip() for l in lf if l.strip()]
            if lines:
                last_rec = json.loads(lines[-1])
                prev_hash = last_rec.get("receipt_sha256", prev_hash)
    record["prev_receipt_sha256"] = prev_hash
    canonical = json.dumps(record, sort_keys=True).encode("utf-8")
    record["receipt_sha256"] = sha256_bytes(canonical)
    with open(LEDGER_PATH, "a") as lf:
        lf.write(json.dumps(record) + "\n")
    return record


def _run_job_subprocess(job_id: str, req: Dict[str, Any]):
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    log_path = os.path.join(job_dir, "worker.log")
    manifest_path = os.path.join(job_dir, "crypto_receipt.json")

    script_rel = req["script"]
    script_abs = os.path.join(REPO_DIR, script_rel)
    args_list: List[str] = [str(a) for a in req.get("args", [])]
    env_vars: Dict[str, str] = dict(os.environ)
    for k, v in req.get("env", {}).items():
        env_vars[str(k)] = str(v)
    env_vars["PYTHONPATH"] = REPO_DIR + os.pathsep + env_vars.get("PYTHONPATH", "")

    cmd = [sys.executable, "-u", script_abs] + args_list
    hw_start = probe_hardware_credentials()

    with JOBS_LOCK:
        JOBS[job_id]["status"] = "RUNNING"
        JOBS[job_id]["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        JOBS[job_id]["hardware_start"] = hw_start
        JOBS[job_id]["peak_vram_mib"] = 0.0
        JOBS[job_id]["peak_host_ram_gb"] = 0.0

    stop_monitor = threading.Event()

    def _telemetry_poller():
        while not stop_monitor.is_set():
            hw = probe_hardware_credentials()
            vram = max([g.get("vram_used_mib", 0.0) for g in hw.get("gpu_devices", [])] or [0.0])
            ram = hw.get("host_ram", {}).get("used_gb", 0.0)
            with JOBS_LOCK:
                if vram > JOBS[job_id]["peak_vram_mib"]:
                    JOBS[job_id]["peak_vram_mib"] = vram
                if ram > JOBS[job_id]["peak_host_ram_gb"]:
                    JOBS[job_id]["peak_host_ram_gb"] = ram
            time.sleep(0.5)

    mon_thread = threading.Thread(target=_telemetry_poller, daemon=True)
    mon_thread.start()

    t0 = time.time()
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_DIR,
                env=env_vars,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with JOBS_LOCK:
                JOBS[job_id]["pid"] = proc.pid
            rc = proc.wait()
        wall_sec = round(time.time() - t0, 3)
        stop_monitor.set()
        mon_thread.join(timeout=2.0)
        hw_end = probe_hardware_credentials()

        # Hash any declared output artifacts
        output_hashes = {}
        for out_rel in req.get("expected_outputs", []):
            out_abs = os.path.join(REPO_DIR, out_rel)
            if os.path.exists(out_abs) and os.path.isfile(out_abs):
                output_hashes[out_rel] = {
                    "sha256": sha256_file(out_abs),
                    "bytes": os.path.getsize(out_abs),
                }

        receipt = append_crypto_ledger({
            "job_id": job_id,
            "milestone": req.get("milestone", "M13_M20"),
            "script": script_rel,
            "script_sha256": sha256_file(script_abs) if os.path.exists(script_abs) else None,
            "args": args_list,
            "return_code": int(rc),
            "wall_clock_seconds": wall_sec,
            "peak_vram_mib": JOBS[job_id]["peak_vram_mib"],
            "peak_host_ram_gb": JOBS[job_id]["peak_host_ram_gb"],
            "hardware_start": hw_start,
            "hardware_end": hw_end,
            "output_artifacts": output_hashes,
            "log_sha256": sha256_file(log_path) if os.path.exists(log_path) else None,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        with open(manifest_path, "w") as mf:
            json.dump(receipt, mf, indent=2)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "COMPLETED" if rc == 0 else "FAILED"
            JOBS[job_id]["return_code"] = int(rc)
            JOBS[job_id]["wall_clock_seconds"] = wall_sec
            JOBS[job_id]["receipt"] = receipt
    except Exception as e:
        stop_monitor.set()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "FAILED"
            JOBS[job_id]["error"] = str(e)


class LeanBridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _check_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1].strip() != API_TOKEN:
            self._send_json(401, {"error": "Unauthorized"})
            return False
        return True

    def _send_json(self, code: int, payload: Dict[str, Any]):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, filepath: str):
        with open(filepath, "rb") as f:
            data = f.read()
        file_sha = sha256_bytes(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
        self.send_header("X-Artifact-SHA256", file_sha)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            return self._send_json(200, {"status": "ok", "service": "lean_30b_colab_worker"})

        if not self._check_auth():
            return

        if path == "/gpu_status":
            return self._send_json(200, probe_hardware_credentials())

        if path.startswith("/job_status/"):
            job_id = path.split("/job_status/", 1)[1].strip("/")
            with JOBS_LOCK:
                if job_id not in JOBS:
                    return self._send_json(404, {"error": f"Unknown job_id {job_id}"})
                info = dict(JOBS[job_id])
            log_path = os.path.join(JOBS_DIR, job_id, "worker.log")
            if os.path.exists(log_path):
                with open(log_path, "r") as lf:
                    info["log_tail"] = [l.rstrip() for l in lf.readlines()[-40:]]
            return self._send_json(200, info)

        if path.startswith("/download_artifact/"):
            job_id = path.split("/download_artifact/", 1)[1].strip("/")
            rel_path = qs.get("path", ["receipt"])[0]
            if rel_path == "receipt":
                p = os.path.join(JOBS_DIR, job_id, "crypto_receipt.json")
            elif rel_path == "log":
                p = os.path.join(JOBS_DIR, job_id, "worker.log")
            else:
                p = os.path.abspath(os.path.join(REPO_DIR, rel_path))
                if not p.startswith(REPO_DIR):
                    return self._send_json(403, {"error": "Invalid artifact path"})
            if not os.path.exists(p):
                return self._send_json(404, {"error": f"Artifact not found: {rel_path}"})
            return self._send_file(p)

        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/upload_file":
            rel_path = body["rel_path"].lstrip("/")
            expected_sha = body["expected_sha256"]
            content = base64.b64decode(body["content_b64"])
            dest = os.path.abspath(os.path.join(REPO_DIR, rel_path))
            if not dest.startswith(REPO_DIR):
                return self._send_json(403, {"error": "Path traversal denied"})
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(content)
            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha:
                os.remove(dest)
                return self._send_json(400, {"error": f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"})
            return self._send_json(200, {"status": "verified", "rel_path": rel_path, "sha256": actual_sha})

        if path == "/sync_bundle":
            expected_sha = body["expected_sha256"]
            tar_bytes = base64.b64decode(body["tar_gz_b64"])
            actual_sha = sha256_bytes(tar_bytes)
            if actual_sha != expected_sha:
                return self._send_json(400, {"error": f"Bundle SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"})
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                tf.extractall(REPO_DIR)
            return self._send_json(200, {"status": "synced", "bundle_sha256": actual_sha})

        if path == "/start_job":
            script_rel = body["script"]
            script_abs = os.path.join(REPO_DIR, script_rel)
            if not os.path.exists(script_abs):
                return self._send_json(404, {"error": f"Script {script_rel} not found in repo"})
            job_id = body.get("job_id") or f"lean_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "job_id": job_id,
                    "request": body,
                    "status": "QUEUED",
                    "queued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            t = threading.Thread(target=_run_job_subprocess, args=(job_id, body), daemon=True)
            t.start()
            return self._send_json(200, {"job_id": job_id, "status": "QUEUED"})

        return self._send_json(404, {"error": "Not found"})


def launch_cloudflared_tunnel(port: int) -> Optional[str]:
    cf_bin = shutil.which("cloudflared")
    if not cf_bin and sys.platform.startswith("linux"):
        cf_dest = "/tmp/cloudflared"
        if not os.path.exists(cf_dest):
            print("[Tunnel] Downloading cloudflared-linux-amd64...", flush=True)
            subprocess.run(
                [
                    "curl", "-fsSL",
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "-o", cf_dest,
                ],
                check=True,
            )
            os.chmod(cf_dest, 0o755)
        cf_bin = cf_dest
    if not cf_bin:
        return None
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pattern = re.compile(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)")
    found_url = []

    def _drain():
        for line in proc.stdout:
            if not found_url:
                m = pattern.search(line)
                if m:
                    found_url.append(m.group(1))

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    t0 = time.time()
    while time.time() - t0 < 30:
        if found_url:
            return found_url[0]
        time.sleep(0.2)
    return None


def main():
    global API_TOKEN, WORKDIR, REPO_DIR, JOBS_DIR, LEDGER_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--token", default=API_TOKEN)
    ap.add_argument("--workdir", default=WORKDIR)
    ap.add_argument("--tunnel", action="store_true")
    args = ap.parse_args()
    API_TOKEN = args.token
    WORKDIR = os.path.abspath(args.workdir)
    REPO_DIR = os.path.join(WORKDIR, "repo")
    JOBS_DIR = os.path.join(WORKDIR, "jobs")
    LEDGER_PATH = os.path.join(WORKDIR, "colab_crypto_ledger.jsonl")
    for _d in (WORKDIR, REPO_DIR, JOBS_DIR):
        os.makedirs(_d, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), LeanBridgeHandler)
    public_url = f"http://{args.host}:{args.port}"
    if args.tunnel:
        cf_url = launch_cloudflared_tunnel(args.port)
        if cf_url:
            public_url = cf_url

    hw_info = probe_hardware_credentials()
    print("=" * 78, flush=True)
    print(f"ANTIGRAVITY_COLAB_ENDPOINT = {public_url}", flush=True)
    print(f"ANTIGRAVITY_COLAB_TOKEN    = {API_TOKEN}", flush=True)
    print(f"GPU_DEVICES                = {json.dumps(hw_info.get('gpu_devices'))}", flush=True)
    print(f"HOST_RAM                   = {json.dumps(hw_info.get('host_ram'))}", flush=True)
    print(f"HARDWARE_SHA256            = {hw_info.get('hardware_fingerprint_sha256')}", flush=True)
    print("=" * 78, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
''')

token = os.environ.get("ANTIGRAVITY_COLAB_TOKEN", "lean-30b-" + secrets.token_hex(8))
print("[3/3] Launching Project LEAN 30B REST GPU Worker + Cloudflare Tunnel...", flush=True)
proc = subprocess.Popen(
    [
        sys.executable,
        "-u",
        str(WORKDIR / "colab_lean_worker_daemon.py"),
        "--port",
        "18765",
        "--token",
        token,
        "--workdir",
        str(WORKDIR),
        "--tunnel",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)

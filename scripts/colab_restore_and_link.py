#!/usr/bin/env python3
"""
scripts/colab_restore_and_link.py
=================================
One-command Google Colab restore & live telemetry tunnel launcher for `Muse-Glimmer-30B`.

Run in any Google Colab GPU cell (T4 / L4 / A100):
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/Mehardeep-Singh-GG/LEAN.git /content/LEAN || (cd /content/LEAN && git pull)
    !python3 /content/LEAN/scripts/colab_restore_and_link.py

It automatically:
  1. Restores the 4 quantized shards (6.53 GB) from `/content/drive/MyDrive/LEAN_30B_BACKUP` into `release_30b/`.
  2. Installs `cloudflared` if not present.
  3. Launches `experiments/start_30b_live_daemon.py` with live `/telemetry`, `/health`, and `/generate_stream` endpoints.
  4. Prints the live `https://*.trycloudflare.com` URL ready to paste into the Studio UI ("Colab Link").
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def ensure_cloudflared() -> Path:
    cf = Path("/tmp/cloudflared")
    if cf.exists():
        return cf
    print("[1/3] Downloading cloudflared tunnel binary...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, str(cf))
    cf.chmod(0o755)
    return cf


def restore_shards_from_drive(repo_root: Path) -> None:
    rel_dir = repo_root / "release_30b"
    existing = list(rel_dir.glob("model_lean_30b-0000*-of-00004.safetensors"))
    if len(existing) == 4:
        print("[2/3] All 4 quantized shards already present in release_30b/.")
        return

    search_roots = [
        Path("/content/drive/MyDrive/LEAN_30B_BACKUP"),
        Path("/content/lean_30b_backing"),
        Path("/content/lean_30b_colab/repo/release_30b"),
    ]
    found_shards = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("model_lean_30b-0000*-of-00004.safetensors"):
            found_shards.setdefault(p.name, p)

    if len(found_shards) < 4:
        raise FileNotFoundError(
            f"Found {len(found_shards)}/4 shards in Google Drive backup. "
            "Ensure Google Drive is mounted at /content/drive and LEAN_30B_BACKUP exists."
        )

    print("[2/3] Restoring 4 quantized Muse-Glimmer-30B shards (6.53 GB) from Google Drive...")
    for name in sorted(found_shards):
        src = found_shards[name]
        dst = rel_dir / name
        if not dst.exists():
            print(f"  -> Copying {name} ({src.stat().st_size / 1e9:.2f} GB)...")
            shutil.copy2(src, dst)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    ensure_cloudflared()
    restore_shards_from_drive(repo_root)

    print("[3/3] Starting 52-Layer Split-K Triton Daemon + Live Hardware Telemetry Tunnel...")
    subprocess.run([sys.executable, str(repo_root / "experiments" / "start_30b_live_daemon.py")], check=True)

    ep_file = repo_root / "release_30b" / "live_infer_endpoint.json"
    if ep_file.exists():
        info = json.loads(ep_file.read_text())
        url = info.get("infer_endpoint", "")
        print("\n" + "=" * 78)
        print("  COLAB GPU TELEMETRY & INFERENCE TUNNEL ACTIVE")
        print(f"  Paste this URL into Spatial Studio -> 'Colab Link':\n  {url}")
        print(f"  Live Telemetry JSON: {url}/telemetry")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

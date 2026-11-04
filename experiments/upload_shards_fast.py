#!/usr/bin/env python3
"""
experiments/upload_shards_fast.py
=================================
Uploads the 4 real 52-layer LEAN-30B `.safetensors` shards from the Colab A100 VM's
2 Gbps connection to Gofile's high-speed CDN so the user's Mac can download them at full
broadband speed (bypassing trycloudflare.com's 70 KB/s quick-tunnel rate limit).
"""

import os
import json
import time
import urllib.request
import subprocess
from pathlib import Path


def get_gofile_server() -> str:
    req = urllib.request.Request(
        "https://api.gofile.io/servers",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    servers = data.get("data", {}).get("servers", [])
    if servers:
        return servers[0]["name"]
    return "store1"


def main():
    out_dir = Path("release_30b")
    shards = [out_dir / f"model_lean_30b-0000{i+1}-of-00004.safetensors" for i in range(4)]
    for s in shards:
        print(f"[Verify] {s}: {s.stat().st_size / (1024**3):.3f} GiB", flush=True)

    server = get_gofile_server()
    print(f"[CDN] Selected Gofile edge server: {server}.gofile.io", flush=True)

    # First create a guest account token + folder so all 4 shards share a token
    upload_url = f"https://{server}.gofile.io/contents/uploadfile"
    results = {"server": server, "files": {}}
    token = None
    folder_id = None

    for idx, s in enumerate(shards, 1):
        t0 = time.perf_counter()
        cmd = ["curl", "-sS", "-X", "POST", upload_url, "-F", f"file=@{s}"]
        if token:
            cmd.extend(["-H", f"Authorization: Bearer {token}"])
        if folder_id:
            cmd.extend(["-F", f"folderId={folder_id}"])
        print(f"[Uploading {idx}/4] {s.name} ({s.stat().st_size / (1024**3):.2f} GiB) -> {upload_url}...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        resp = json.loads(proc.stdout)
        dt = time.perf_counter() - t0
        if resp.get("status") == "ok":
            d = resp["data"]
            if not token and "guestToken" in d:
                token = d["guestToken"]
            if not folder_id and "parentFolder" in d:
                folder_id = d["parentFolder"]
            results["files"][s.name] = d
            results["guestToken"] = token
            results["folderId"] = folder_id
            print(f"  [Uploaded {s.name} in {dt:.1f}s] id={d.get('id')} page={d.get('downloadPage')}", flush=True)
        else:
            print(f"  [Error uploading {s.name}]: {proc.stdout}", flush=True)

    with open(out_dir / "cdn_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[SUCCESS] Saved release_30b/cdn_manifest.json!", flush=True)


if __name__ == "__main__":
    main()

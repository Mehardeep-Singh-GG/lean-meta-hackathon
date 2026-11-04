#!/usr/bin/env python3
"""
experiments/download_30b_shards_to_mac.py
=========================================
Downloads the real 52-layer LEAN-quantized `Muse-Glimmer-30B` shards from the Colab Pro+ A100 worker
by streaming 48 MiB chunks in parallel and writing each chunk directly to its exact byte offset
in `release_30b/model_lean_30b-00001-of-00004.safetensors` .. `00004` (zero extra SSD overhead).
"""

import os
import sys
import json
import time
import shutil
import argparse
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_chunk_bytes(endpoint: str, token: str, job_id: str, remote_relpath: str, retries: int = 4) -> bytes:
    qs = urllib.parse.urlencode({"path": remote_relpath})
    url = f"{endpoint.rstrip('/')}/download_artifact/{job_id}?{qs}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "ProjectLEAN-AntigravityBridge/M13",
    }
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {remote_relpath} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path("release_30b")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("release_30b_chunks/manifest.json")
    if not manifest_path.exists():
        raw = fetch_chunk_bytes(args.endpoint, args.token, args.job_id, "release_30b_chunks/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(raw)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Copy small config/tokenizer files if not already in release_30b/
    for sf in manifest.get("small_files", []):
        dest = out_dir / sf
        if not dest.exists() or dest.stat().st_size == 0:
            raw = fetch_chunk_bytes(args.endpoint, args.token, args.job_id, f"release_30b_chunks/{sf}")
            dest.write_bytes(raw)
            print(f"[Config] Saved {dest} ({len(raw):,} bytes)", flush=True)

    # Also copy bm25_index.pkl from release/ if available
    bm25_src = Path("release/bm25_index.pkl")
    if bm25_src.exists() and not (out_dir / "bm25_index.pkl").exists():
        shutil.copy2(bm25_src, out_dir / "bm25_index.pkl")

    t0 = time.perf_counter()
    total_all_bytes = sum(v["total_size"] for v in manifest["shards"].values())
    print(f"[Download] Starting parallel download of 4 shards ({total_all_bytes / (1024**3):.3f} GiB total)...", flush=True)

    downloaded_bytes = 0
    lock = threading.Lock()

    for s_name, s_info in sorted(manifest["shards"].items()):
        s_path = out_dir / s_name
        total_sz = int(s_info["total_size"])
        parts = s_info["parts"]

        if s_path.exists() and s_path.stat().st_size == total_sz:
            print(f"  [Skip] {s_name} already exists with exact size ({total_sz / (1024**3):.3f} GiB)", flush=True)
            downloaded_bytes += total_sz
            continue

        # Pre-allocate file size and write chunks sequentially or by offset
        with open(s_path, "wb") as wf:
            wf.truncate(total_sz)

        file_lock = threading.Lock()
        offsets = []
        cur = 0
        for p in parts:
            offsets.append((p["name"], cur, int(p["size"])))
            cur += int(p["size"])

        def _dl_and_write(item):
            nonlocal downloaded_bytes
            p_name, offset, sz = item
            data = fetch_chunk_bytes(args.endpoint, args.token, args.job_id, f"release_30b_chunks/{p_name}")
            if len(data) != sz:
                raise RuntimeError(f"Chunk {p_name} size mismatch: expected {sz}, got {len(data)}")
            with file_lock:
                with open(s_path, "r+b") as wf:
                    wf.seek(offset)
                    wf.write(data)
            with lock:
                downloaded_bytes += sz
                elapsed = max(1e-3, time.perf_counter() - t0)
                mbps = (downloaded_bytes / (1024**2)) / elapsed
                pct = 100.0 * downloaded_bytes / total_all_bytes
                print(
                    f"  [{pct:5.1f}%] {s_name} <- {p_name} ({downloaded_bytes / (1024**3):.2f} / {total_all_bytes / (1024**3):.2f} GiB @ {mbps:.1f} MiB/s)",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_dl_and_write, it) for it in offsets]
            for fut in as_completed(futs):
                fut.result()

        print(f"[Shard Complete] {s_path} ({s_path.stat().st_size / (1024**3):.3f} GiB)", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n[SUCCESS] Downloaded all 4 shards ({total_all_bytes / (1024**3):.3f} GiB) to {out_dir}/ in {dt:.1f}s!", flush=True)


if __name__ == "__main__":
    main()

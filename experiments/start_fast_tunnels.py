#!/usr/bin/env python3
"""
experiments/start_fast_tunnels.py
=================================
Spawns persistent, detached TCP tunnels (cloudflared --protocol http2 and localhost.run SSH)
pointing to 127.0.0.1:18765 on the Colab A100 worker so that downloading the 4 shards to Mac
uses full TCP window scaling without Docker's 208 KB UDP/QUIC buffer throttle.
"""

import os
import re
import json
import time
import subprocess
from pathlib import Path


def main():
    out_dir = Path("release_30b")
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = {}

    # 1. Detached cloudflared with --protocol http2 (TCP instead of UDP/QUIC)
    cf_log = Path("/tmp/cf_tcp.log")
    if cf_log.exists():
        cf_log.unlink()
    with open(cf_log, "w") as lf:
        subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:18765"],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    # 2. Detached localhost.run SSH TCP tunnel
    lhr_log = Path("/tmp/lhr_tcp.log")
    if lhr_log.exists():
        lhr_log.unlink()
    if not Path("/root/.ssh/id_rsa").exists():
        Path("/root/.ssh").mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "rsa", "-b", "2048", "-f", "/root/.ssh/id_rsa", "-N", ""], check=False)
    with open(lhr_log, "w") as lf:
        subprocess.Popen(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ServerAliveInterval=20",
                "-R", "80:127.0.0.1:18765",
                "nokey@localhost.run",
            ],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    t0 = time.time()
    while time.time() - t0 < 15.0:
        if cf_log.exists():
            txt = cf_log.read_text(errors="ignore")
            m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", txt)
            if m:
                urls["cf_tcp"] = m.group(1)
        if lhr_log.exists():
            txt = lhr_log.read_text(errors="ignore")
            m = re.search(r"(https://[a-zA-Z0-9\-]+\.lhr\.life)", txt)
            if m:
                urls["lhr_tcp"] = m.group(1)
        if "cf_tcp" in urls and "lhr_tcp" in urls:
            break
        time.sleep(0.5)

    print(f"[Fast TCP Tunnels] {json.dumps(urls, indent=2)}", flush=True)
    with open(out_dir / "fast_tunnels.json", "w") as f:
        json.dump(urls, f, indent=2)


if __name__ == "__main__":
    main()

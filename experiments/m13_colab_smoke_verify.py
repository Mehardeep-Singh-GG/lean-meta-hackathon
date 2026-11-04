# File: experiments/m13_colab_smoke_verify.py
"""M13 End-to-End Hardware Telemetry & Artifact Provenance Smoke Experiment."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    out_dir = Path("results/m13_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "milestone": "M13",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_available": False,
        "device_name": "CPU",
        "allocated_mib": 0.0,
        "matmul_checksum": 0.0,
    }
    try:
        import torch
        metrics["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            metrics["cuda_available"] = True
            metrics["device_name"] = torch.cuda.get_device_name(0)
            torch.manual_seed(42)
            a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
            b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            metrics["allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
            metrics["matmul_checksum"] = float(c.float().abs().mean().item())
        else:
            torch.manual_seed(42)
            a = torch.randn((512, 512), dtype=torch.float32)
            metrics["matmul_checksum"] = float((a @ a.T).abs().mean().item())
    except Exception as exc:
        metrics["error"] = str(exc)

    # Hold execution for 1.6s so 2Hz telemetry poller captures >= 3 samples
    time.sleep(1.6)
    out_file = out_dir / "smoke_metrics.json"
    out_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[M13-SMOKE] Wrote verified metrics to {out_file}: checksum={metrics['matmul_checksum']:.6f}", flush=True)


if __name__ == "__main__":
    main()

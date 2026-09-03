"""Benchmark GPU: mide throughput de la CNN en CPU vs tu 9060 XT (ROCm).

Uso:  python scripts/benchmark_gpu.py [--batch 32] [--iters 50]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # root del proyecto

import torch
from src.models.cnn import build_model
from src.utils.device import get_device, device_info


def bench(device_str: str, backbone: str, batch: int, iters: int, warmup: int = 5) -> float:
    device = get_device(device_str)
    model = build_model(backbone).to(device).eval()
    x = torch.randn(batch, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return batch * iters / dt  # imgs/seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--backbone", default="simple_cnn",
                    choices=["simple_cnn", "efficientnet_b0"])
    args = ap.parse_args()
    print(device_info())
    cpu = bench("cpu", args.backbone, args.batch, args.iters)
    print(f"CPU : {cpu:7.1f} imgs/s")
    if torch.cuda.is_available():
        gpu = bench("cuda", args.backbone, args.batch, args.iters)
        print(f"GPU : {gpu:7.1f} imgs/s  (speedup x{gpu / cpu:.1f})")
    else:
        print("GPU : no disponible")


if __name__ == "__main__":
    main()

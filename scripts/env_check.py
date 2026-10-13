"""Verify the GPU environment for the deterministic_model project.

Checks: PyTorch + CUDA versions, every visible GPU (name, VRAM, compute
capability), whether this PyTorch build ships kernels for each card's
architecture (Blackwell sm_120 needs CUDA 12.8+), bf16 support, and a real
matmul on each device.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch not installed. See README quick start.")
        return 1

    ok = True
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available to PyTorch.")
        return 1

    arch_list = torch.cuda.get_arch_list()
    print(f"Arch list   : {' '.join(arch_list)}")
    print(f"GPU count   : {torch.cuda.device_count()}\n")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cc = f"{props.major}{props.minor}"
        vram = props.total_memory / 1024**3
        print(f"[cuda:{i}] {props.name}")
        print(f"  VRAM        : {vram:.1f} GB")
        print(f"  Compute cap : sm_{cc}")

        if f"sm_{cc}" not in arch_list:
            print(f"  [WARN] This PyTorch build has no sm_{cc} kernels.")
            if props.major >= 12:
                print("         Blackwell needs: pip install torch --index-url "
                      "https://download.pytorch.org/whl/cu128")
            ok = False

        print(f"  bf16        : {torch.cuda.is_bf16_supported()}")

        try:
            dev = torch.device(f"cuda:{i}")
            a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
            b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
            for _ in range(5):  # warmup: cuBLAS init + kernel selection
                a @ b
            torch.cuda.synchronize(dev)
            # events must be recorded on this device's stream, not cuda:0's
            with torch.cuda.device(dev):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    a @ b
                end.record()
            torch.cuda.synchronize(dev)
            ms = start.elapsed_time(end) / 20
            tflops = 2 * 2048**3 / (ms / 1000) / 1e12
            print(f"  matmul test : OK ({ms:.2f} ms, ~{tflops:.1f} TFLOPS bf16)")
        except Exception as e:  # noqa: BLE001
            print(f"  matmul test : FAIL ({e})")
            ok = False
        print()

    print("Recommendation: set train_device to the 3090 Ti index and "
          "eval_device to the 5060 Ti index in configs/phase1.yaml.")
    print("RESULT:", "ALL GOOD" if ok else "ISSUES FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

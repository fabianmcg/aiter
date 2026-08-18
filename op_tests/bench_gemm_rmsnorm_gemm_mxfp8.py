# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import statistics

import torch
import torch.profiler

import aiter
from aiter import dtypes
from aiter.ops.gemm_rmsnorm_gemm import (
    quantize_mxfp8_gfx950,
    quantize_mxfp8_weight_nhid,
)
from aiter.ops.quant import per_1x32_mx_quant_hip

from typing import Callable


def profile(fn: Callable[[], []], iters: int, warmup: int):
    """Median per-call latency (ms) and last output."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(iters):
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return (
        statistics.median(times),
        out,
        str(
            prof.key_averages(group_by_input_shape=False).table(
                sort_by="self_cuda_time_total",
                row_limit=25,
            )
        ),
    )


def quant_1x32(x: torch.tensor):
    return per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)


def make_inputs(m: int, nHid: int, k1: int, nOut: int) -> tuple[torch.Tensor, ...]:
    a = (torch.randn(m, k1, device="cuda") * 0.1).to(torch.bfloat16)
    b1 = (torch.randn(nHid, k1, device="cuda") * 0.1).to(torch.bfloat16)
    gamma = (torch.rand(nHid, device="cuda") + 0.5).to(torch.bfloat16)
    w2 = (torch.randn(nOut, nHid, device="cuda") * 0.1).to(torch.bfloat16)
    residual = (torch.randn(m, nHid, device="cuda") * 0.1).to(torch.bfloat16)
    return a, b1, gamma, w2, residual


def make_paths(m: int, nHid: int, k1: int, nOut: int, eps: float):
    """Return {name: callable}; every callable returns (out, hidden) where hidden
    is the pre-RMSNorm GEMM1 result H = A @ W1."""
    a, b1, gamma, w2, residual = make_inputs(m, nHid, k1, nOut)

    b2Fused, scaleB2Fused = quantize_mxfp8_weight_nhid(w2)
    aFused, scaleAFused = quantize_mxfp8_gfx950(a)
    b1Fused, scaleB1Fused = quantize_mxfp8_gfx950(b1)

    aU, sA1 = quant_1x32(a)
    b1U, sB1 = quant_1x32(b1)
    b2U, sB2 = quant_1x32(w2)

    def fusedBf16():
        out, _, hidden = aiter.gemm_rmsnorm_gemm_mxfp8(
            a,
            b1,
            gamma,
            b2Fused,
            scaleB2Fused,
            eps,
            return_residual=True,
            residual=residual,
        )
        return out, hidden

    def unfusedBf16():
        hidden = torch.mm(a, b1.t(), out_dtype=torch.bfloat16) + residual
        normed = aiter.rmsnorm2d_fwd(hidden, gamma, eps)
        a2, sA2 = quant_1x32(normed)
        out = torch._scaled_mm(
            a2, b2U.t(), scale_a=sA2, scale_b=sB2, out_dtype=torch.bfloat16
        )
        return out, hidden

    def fusedFp8():
        out, _, hidden = aiter.gemm_rmsnorm_gemm_mxfp8_fp8in(
            aFused,
            scaleAFused,
            b1Fused,
            scaleB1Fused,
            gamma,
            b2Fused,
            scaleB2Fused,
            eps,
            return_residual=True,
            residual=residual,
        )
        return out, hidden

    def unfusedFp8():
        hidden = (
            torch._scaled_mm(
                aU, b1U.t(), scale_a=sA1, scale_b=sB1, out_dtype=torch.bfloat16
            )
            + residual
        )
        normed = aiter.rmsnorm2d_fwd(hidden, gamma, eps)
        a2, sA2 = quant_1x32(normed)
        out = torch._scaled_mm(
            a2, b2U.t(), scale_a=sA2, scale_b=sB2, out_dtype=torch.bfloat16
        )
        return out, hidden

    return {
        "bf16_fused": fusedBf16,
        "bf16_unfused": unfusedBf16,
        "fp8_fused": fusedFp8,
        "fp8_unfused": unfusedFp8,
    }


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def run_shape(m: int, nHid: int, k1: int, nOut: int, warmup: int = 10, iters: int = 20):
    torch.manual_seed(0)
    paths = make_paths(m, nHid, k1, nOut, eps=1e-5)

    print("*" * 80)
    print(f"Shape: m = {m}, nHid = {nHid}, k1 = {k1}, nOut = {nOut}")

    results = {}
    for name, fn in paths.items():
        t, (out, hidden), table = profile(fn, iters, warmup)
        results[name] = (t, out, hidden, table)

    def report(fused_key: str, unfused_key: str, label: str):
        t_f, out_f, hidden_f, table_f = results[fused_key]
        t_u, out_u, hidden_u, table_u = results[unfused_key]
        speedup = t_u / t_f
        out_err = relative_error(out_f, out_u)
        hidden_err = relative_error(hidden_f, hidden_u)
        print(f"\n{label}:")
        print(
            f"  fused: {t_f:.3f} ms  unfused: {t_u:.3f} ms  speedup: {speedup:.2f}x  "
            f"out rel_err: {out_err:.3e}  hidden rel_err: {hidden_err:.3e}"
        )
        print(f"Fused profile:\n{table_f}")
        print(f"Unfused profile:\n{table_u}")

    report("bf16_fused", "bf16_unfused", "BF16")
    report("fp8_fused", "fp8_unfused", "FP8-input")
    print("*" * 80)


if __name__ == "__main__":
    shapes = [
        (1024, 2048, 4096, 4096),
        (8192, 8192, 8192, 8192),
    ]

    [run_shape(*shape) for shape in shapes]

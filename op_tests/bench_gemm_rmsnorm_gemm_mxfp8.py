# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import time

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


def profile(fn: Callable[[int, bool], tuple], iters: int, warmup: int):
    """Profiler table over iters perf runs. No wall/event timing is taken; read
    timing from the returned table. Callers do the numeric error check separately."""
    torch.cuda.synchronize()
    for i in range(warmup):
        fn(i, False)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for i in range(iters):
            fn(i, False)
            torch.cuda.synchronize()
            time.sleep(0.3)
    return str(
        prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cuda_time_total",
            row_limit=25,
        )
    )


def quant_1x32(x: torch.tensor):
    return per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)


def _torchRmsnorm(x: torch.Tensor, g: torch.Tensor, e: float) -> torch.Tensor:
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + e)
    return (x_fp32 * rstd).to(x.dtype) * g


_compiledRmsnorm = torch.compile(_torchRmsnorm)


def make_inputs(m: int, nHid: int, k1: int, nOut: int) -> tuple[torch.Tensor, ...]:
    a = (torch.randn(m, k1, device="cuda") * 0.1).to(torch.bfloat16)
    b1 = (torch.randn(nHid, k1, device="cuda") * 0.1).to(torch.bfloat16)
    gamma = (torch.rand(nHid, device="cuda") + 0.5).to(torch.bfloat16)
    w2 = (torch.randn(nOut, nHid, device="cuda") * 0.1).to(torch.bfloat16)
    residual = (torch.randn(m, nHid, device="cuda") * 0.1).to(torch.bfloat16)
    return a, b1, gamma, w2, residual


def make_paths(
    m: int, nHid: int, k1: int, nOut: int, eps: float, family: str, rotations: int = 5
):
    """Return {name: callable(iteration, check)}; every callable returns
    (out, hidden) where hidden is the pre-RMSNorm GEMM1 result H = A @ W1 +
    residual. Inputs and output buffers are pooled into `rotations` independent
    sets; each call uses set `iteration % rotations` so successive perf
    iterations touch cold memory."""
    inputs = [make_inputs(m, nHid, k1, nOut) for _ in range(rotations)]

    if family == "bf16":
        # Rotating per-path output buffers; each hidden buffer doubles as the
        # addmm addend for in-place accumulation, so it starts as a copy of the
        # matching residual.
        hiddenAiterPool = [residual.clone() for _, _, _, _, residual in inputs]
        outAiterPool = [
            torch.empty(m, nOut, device="cuda", dtype=torch.bfloat16)
            for _ in range(rotations)
        ]
        hiddenTorchPool = [residual.clone() for _, _, _, _, residual in inputs]
        outTorchPool = [
            torch.empty(m, nOut, device="cuda", dtype=torch.bfloat16)
            for _ in range(rotations)
        ]
        # Transposed weight views are metadata-only and constant across
        # iterations; build them once so the perf loop makes no
        # aten::t/transpose/as_strided calls.
        weightViewsBf16 = [(b1.t(), w2.t()) for _, b1, _, w2, _ in inputs]

        def fusedBf16(iteration: int, check: bool):
            a, b1, gamma, w2, residual = inputs[iteration % rotations]
            out, hidden = aiter.gemm_rmsnorm_gemm_bf16(
                a, b1, gamma, w2, eps, residual_in=residual
            )
            return out, hidden

        def unfusedBf16(iteration: int, check: bool):
            # In-place addmm with C == out == hidden avoids the residual copy
            # (no __amd_rocclr_copyBuffer). Reset to residual only for the error-
            # check run; perf iterations accumulate, which is fine for throughput.
            idx = iteration % rotations
            a, _, gamma, _, residual = inputs[idx]
            b1t, w2t = weightViewsBf16[idx]
            hidden = hiddenAiterPool[idx]
            out = outAiterPool[idx]
            if check:
                hidden.copy_(residual)
            torch.addmm(hidden, a, b1t, out=hidden)
            normed = aiter.rmsnorm2d_fwd(hidden, gamma, eps)
            torch.mm(normed, w2t, out=out)
            return out, hidden

        def unfusedBf16Torch(iteration: int, check: bool):
            # Same in-place trick; residual is the addmm addend so the compiled
            # rmsnorm is a pure norm. Reset only for the error-check run.
            idx = iteration % rotations
            a, _, gamma, _, residual = inputs[idx]
            b1t, w2t = weightViewsBf16[idx]
            hidden = hiddenTorchPool[idx]
            out = outTorchPool[idx]
            if check:
                hidden.copy_(residual)
            torch.addmm(hidden, a, b1t, out=hidden)
            normed = _compiledRmsnorm(hidden, gamma, eps)
            torch.mm(normed, w2t, out=out)
            return out, hidden

        return {
            "bf16_fused": fusedBf16,
            "bf16_unfused": unfusedBf16,
            "bf16_unfused_torch": unfusedBf16Torch,
        }

    # family == "fp8": pre-quantize each rotation's inputs.
    fusedQuant = []
    unfusedQuant = []
    for a, b1, gamma, w2, residual in inputs:
        b2Fused, scaleB2Fused = quantize_mxfp8_weight_nhid(w2)
        aFused, scaleAFused = quantize_mxfp8_gfx950(a)
        b1Fused, scaleB1Fused = quantize_mxfp8_gfx950(b1)
        fusedQuant.append(
            (aFused, scaleAFused, b1Fused, scaleB1Fused, b2Fused, scaleB2Fused)
        )
        aU, sA1 = quant_1x32(a)
        b1U, sB1 = quant_1x32(b1)
        b2U, sB2 = quant_1x32(w2)
        # Cache transposed quantized-weight views (metadata-only, constant).
        unfusedQuant.append((aU, sA1, b1U.t(), sB1, b2U.t(), sB2))

    def fusedFp8(iteration: int, check: bool):
        idx = iteration % rotations
        _, _, gamma, _, residual = inputs[idx]
        aFused, scaleAFused, b1Fused, scaleB1Fused, b2Fused, scaleB2Fused = fusedQuant[
            idx
        ]
        out, _, hidden = aiter.gemm_rmsnorm_gemm_mxfp8(
            aFused,
            scaleAFused,
            b1Fused,
            scaleB1Fused,
            gamma,
            b2Fused,
            scaleB2Fused,
            eps,
            residual=residual,
        )
        return out, hidden

    def unfusedFp8(iteration: int, check: bool):
        idx = iteration % rotations
        _, _, gamma, _, residual = inputs[idx]
        aU, sA1, b1Ut, sB1, b2Ut, sB2 = unfusedQuant[idx]
        hidden = (
            torch._scaled_mm(
                aU, b1Ut, scale_a=sA1, scale_b=sB1, out_dtype=torch.bfloat16
            )
            + residual
        )
        normed = _compiledRmsnorm(hidden, gamma, eps)
        a2, sA2 = quant_1x32(normed)
        out = torch._scaled_mm(
            a2, b2Ut, scale_a=sA2, scale_b=sB2, out_dtype=torch.bfloat16
        )
        return out, hidden

    return {"fp8_fused": fusedFp8, "fp8_unfused": unfusedFp8}


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def run_shape(
    m: int,
    nHid: int,
    k1: int,
    nOut: int,
    family: str,
    warmup: int = 10,
    iters: int = 20,
    rotations: int = 5,
):
    torch.manual_seed(0)
    paths = make_paths(
        m, nHid, k1, nOut, eps=1e-5, family=family, rotations=rotations
    )

    print("*" * 80)
    print(f"Shape: m = {m}, nHid = {nHid}, k1 = {k1}, nOut = {nOut}  family = {family}")

    results = {}
    for name, fn in paths.items():
        try:
            out, hidden = fn(0, True)  # numeric error check: reset -> valid result
            out, hidden = out.clone(), hidden.clone()
            table = profile(fn, iters, warmup)
            results[name] = (out, hidden, table)
        except RuntimeError as exc:
            print(f"  {name}: UNAVAILABLE ({exc}) — skipping shape")
            print("*" * 80)
            return

    def report(fused_key: str, unfused_key: str, label: str):
        out_f, hidden_f, table_f = results[fused_key]
        out_u, hidden_u, table_u = results[unfused_key]
        out_err = relative_error(out_f, out_u)
        hidden_err = relative_error(hidden_f, hidden_u)
        print(f"\n{label}:")
        print(f"  out rel_err: {out_err:.3e}  hidden rel_err: {hidden_err:.3e}")
        print(f"Fused profile:\n{table_f}")
        print(f"Unfused profile:\n{table_u}")

    if family == "bf16":
        report("bf16_fused", "bf16_unfused", "BF16 (aiter rmsnorm)")
        report("bf16_fused", "bf16_unfused_torch", "BF16 (torch.compile rmsnorm)")
    else:
        report("fp8_fused", "fp8_unfused", "FP8-input")
    print("*" * 80)


if __name__ == "__main__":
    # (m, nHid, k1, nOut)
    bf16_shapes = [
        (16384, 2048, 2048, 11264),
        # (8192, 8192, 8192, 8192),
        # (2048, 8192, 2048, 2048),
    ]
    fp8_shapes = [
        (8192, 8192, 8192, 8192),
    ]
    for shape in bf16_shapes:
        run_shape(*shape, family="bf16")
    # for shape in fp8_shapes:
    #     run_shape(*shape, family="fp8")

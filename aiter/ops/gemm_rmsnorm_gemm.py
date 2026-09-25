# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
from collections import namedtuple

import numpy as np
import torch
from torch import Tensor

from ..jit.core import compile_ops, get_gfx

__all__ = [
    "gemm_rmsnorm_gemm_bf16",
    "gemm_rmsnorm_gemm_mxfp8",
    "quantize_mxfp8_gfx950",
    "quantize_mxfp8_weight_nhid",
    "gemm_rmsnorm_gemm",
    "GemmRmsNormGemmOutput",
]


def _require_gfx950():
    """Guard: the custom hipblaslt fused epilogue is gfx950-only."""
    arch = get_gfx()
    if arch != "gfx950":
        raise NotImplementedError(
            f"gemm_rmsnorm_gemm requires gfx950; current arch is {arch!r}"
        )


def _e8m0(amax: float):
    """Return (sb, qmult) for the given amax; mirrors e8m0QuantMult from the reference C++."""
    if amax == 0.0:
        return 0, 0.0
    if not np.isfinite(amax):
        raise ValueError(f"_e8m0: amax must be finite, got {amax}")
    fp8_max   = 448.0
    scale_f   = np.float32(amax / fp8_max)
    bits      = int(scale_f.view(np.uint32))
    exp_byte  = (bits >> 23) & 0xFF
    mant      = bits & 0x7FFFFF
    ceil_adj  = 1 if mant != 0 else 0
    sb        = min(exp_byte + ceil_adj, 254)
    q_exp     = max(1, min(254, 254 - sb))
    q_bits    = np.uint32(q_exp << 23)
    qmult     = float(q_bits.view(np.float32))
    return sb, qmult


def _swizzle_gfx950_np(scale_plain: np.ndarray, padded_rows: int, padded_cols: int) -> np.ndarray:
    """Apply GFX950 pre-swizzle; mirrors swizzleGfx950 from the reference C++.

    scale_plain must be shape (padded_rows, padded_cols).
    """
    col_blocks = padded_cols // 8
    ti, tj = np.meshgrid(np.arange(padded_rows, dtype=np.int64),
                         np.arange(padded_cols, dtype=np.int64), indexing='ij')
    d0 = ti >> 5;  d1 = (ti >> 4) & 1;  d2 = ti & 0xF
    d3 = tj >> 3;  d4 = (tj >> 2) & 1;  d5 = tj & 3
    swz = d0 * (col_blocks * 256) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1
    out = np.zeros(padded_rows * padded_cols, dtype=np.uint8)
    out[swz.ravel()] = scale_plain.ravel()
    return out


def _unswizzle_gfx950_np(swizzled: np.ndarray, padded_rows: int, padded_cols: int) -> np.ndarray:
    """Inverse of swizzleGfx950: swizzled bytes → plain[padded_rows, padded_cols].

    Returns a 2D array of shape (padded_rows, padded_cols).
    """
    col_blocks = padded_cols // 8
    ti, tj = np.meshgrid(np.arange(padded_rows, dtype=np.int64),
                         np.arange(padded_cols, dtype=np.int64), indexing='ij')
    d0 = ti >> 5;  d1 = (ti >> 4) & 1;  d2 = ti & 0xF
    d3 = tj >> 3;  d4 = (tj >> 2) & 1;  d5 = tj & 3
    swz = d0 * (col_blocks * 256) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1
    plain = np.empty(padded_rows * padded_cols, dtype=np.uint8)
    plain[ti.ravel() * padded_cols + tj.ravel()] = swizzled[swz.ravel()]
    return plain.reshape(padded_rows, padded_cols)


def quantize_mxfp8_gfx950(W_bf16: Tensor):
    """Quantize W [nRows, nCols] bf16 to fp8 + GFX950-swizzled UE8M0 block-32 scale.

    Blocks of 32 are taken along nCols for each row.
    Returns (W_fp8 [nRows, nCols] float8_e4m3fn, scale uint8 swizzled).
    """
    nRows, nCols = W_bf16.shape
    W_f32        = W_bf16.float().cpu().numpy()
    block_size   = 32
    nCol_tiles   = (nCols + block_size - 1) // block_size
    padded_rows  = ((nRows + 31) // 32) * 32
    padded_cols  = ((nCol_tiles + 7) // 8) * 8

    W_f32_out   = np.zeros((nRows, nCols), dtype=np.float32)
    scale_plain = np.zeros((padded_rows, padded_cols), dtype=np.uint8)

    for r in range(nRows):
        for cb in range(nCol_tiles):
            c_start = cb * block_size
            c_end   = min(c_start + block_size, nCols)
            amax    = float(np.max(np.abs(W_f32[r, c_start:c_end])))
            sb, qmult = _e8m0(amax)
            scale_plain[r, cb]          = sb
            W_f32_out[r, c_start:c_end] = W_f32[r, c_start:c_end] * qmult

    W_fp8    = torch.from_numpy(W_f32_out).to(W_bf16.device).to(torch.float8_e4m3fn)
    swizzled = _swizzle_gfx950_np(scale_plain, padded_rows, padded_cols)
    scale    = torch.from_numpy(swizzled).to(W_bf16.device)
    return W_fp8, scale


def quantize_mxfp8_weight_nhid(W2_bf16: Tensor):
    """Quantize W2 [nOut, nHid] bf16 to fp8 + GFX950-swizzled UE8M0 scale for consumer B.

    Blocks of 32 are taken along the nHid axis for each output row.
    Returns (B2_fp8 [nOut, nHid] float8_e4m3fn, scaleB2 uint8 swizzled).
    """
    return quantize_mxfp8_gfx950(W2_bf16)


@compile_ops("module_gemm_rmsnorm_gemm", fc_name="gemm_rmsnorm_gemm_bf16", ffi_type="pybind")
def _gemm_rmsnorm_gemm_bf16(
    A: Tensor, W1: Tensor, gamma: Tensor, W2: Tensor,
    eps: float, residual_in: Tensor,
) -> list[Tensor]: ...


def gemm_rmsnorm_gemm_bf16(
    A: Tensor, W1: Tensor, gamma: Tensor, W2: Tensor,
    eps: float = 1e-5, *, residual_in: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused bf16 GEMM1 + RMSNorm + GEMM2.

    A: [M, K1], W1: [N_hidden, K1], gamma: [N_hidden], W2: [N_out, N_hidden].
    residual_in: [M, N_hidden] bf16 added to GEMM1 output before RMSNorm (mandatory).
    Returns (out [M, N_out] bf16, residual_out [M, N_hidden] bf16) where
    residual_out = A @ W1 + residual_in stored as bf16.
    """
    _require_gfx950()
    outs = _gemm_rmsnorm_gemm_bf16(A, W1, gamma, W2, eps, residual_in)
    return outs[0], outs[1]


@compile_ops("module_gemm_rmsnorm_gemm", fc_name="gemm_rmsnorm_gemm_mxfp8", ffi_type="pybind")
def _gemm_rmsnorm_gemm_mxfp8(
    A_fp8: Tensor, scaleA: Tensor,
    B1_fp8: Tensor, scaleB1: Tensor,
    gamma: Tensor,
    B2: Tensor, scaleB2: Tensor,
    eps: float, residual: Tensor,
) -> list[Tensor]: ...


def gemm_rmsnorm_gemm_mxfp8(
    A_fp8: Tensor, scaleA: Tensor,
    B1_fp8: Tensor, scaleB1: Tensor,
    gamma: Tensor,
    B2: Tensor, scaleB2: Tensor,
    eps: float = 1e-5, *, residual: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Full MXFP8 chain: fp8 GEMM1 (MX block-32 input scales) + RMSNorm + fp8 GEMM2.

    A_fp8: [mTok, K1] fp8 e4m3, scaleA: uint8 GFX950-swizzled UE8M0 input-A scale,
    B1_fp8: [nHid, K1] fp8 e4m3, scaleB1: uint8 GFX950-swizzled UE8M0 input-B1 scale,
    gamma: [nHid] bf16, B2: [nOut, nHid] fp8 e4m3,
    scaleB2: uint8 GFX950-swizzled UE8M0 consumer-B scale.
    residual: [mTok, nHid] bf16 mandatory residual added before RMSNorm.
    Returns (out [mTok, nOut] bf16, scaleA2 uint8 UE8M0, residual_out [mTok, nHid] bf16).
    """
    _require_gfx950()
    outs = _gemm_rmsnorm_gemm_mxfp8(
        A_fp8, scaleA, B1_fp8, scaleB1, gamma, B2, scaleB2, eps, residual
    )
    return outs[0], outs[1], outs[2]


GemmRmsNormGemmOutput = namedtuple(
    "GemmRmsNormGemmOutput", ["out", "scaleA", "residual"]
)


def gemm_rmsnorm_gemm(
    A: Tensor,
    B1: Tensor,
    gamma: Tensor,
    B2: Tensor,
    *,
    scaleA: Tensor | None = None,
    scaleB1: Tensor | None = None,
    scaleB2: Tensor | None = None,
    eps: float = 1e-5,
    residual: Tensor,
) -> GemmRmsNormGemmOutput:
    """Unified GEMM->RMSNorm->GEMM dispatcher.

    Operand contract:
      A:       [mTok, K1]    bf16 or float8_e4m3fn
      B1:      [nHid, K1]    bf16 or float8_e4m3fn
      gamma:   [nHid]        bf16
      B2:      [nOut, nHid]  bf16 or float8_e4m3fn
      scaleA/scaleB1/scaleB2: uint8 GFX950 UE8M0 block-32 scales for fp8 operands; None for bf16.
      residual: [mTok, nHid] bf16, mandatory residual added before RMSNorm.

    Dispatch:
      bf16:   A=bf16, B1=bf16, B2=bf16 -> bf16 GEMM1 -> bf16 GEMM2 (no scales)
      mxfp8:  A=fp8,  B1=fp8,  B2=fp8  -> fp8 GEMM1  -> fp8 GEMM2  (requires scaleA, scaleB1, scaleB2)
    """
    _require_gfx950()
    bf16 = torch.bfloat16
    fp8  = torch.float8_e4m3fn

    if A.dtype == bf16 and B1.dtype == bf16 and B2.dtype == bf16:
        if scaleA is not None or scaleB1 is not None or scaleB2 is not None:
            raise ValueError("bf16 chain takes no scales; got a non-None scale")
        out, residual_out = gemm_rmsnorm_gemm_bf16(
            A, B1, gamma, B2, eps, residual_in=residual
        )
        return GemmRmsNormGemmOutput(out=out, scaleA=None, residual=residual_out)

    if A.dtype == fp8 and B1.dtype == fp8 and B2.dtype == fp8:
        if scaleA is None or scaleB1 is None or scaleB2 is None:
            raise ValueError("fp8-in chain requires scaleA, scaleB1, scaleB2")
        out, scaleA2, residual_out = gemm_rmsnorm_gemm_mxfp8(
            A, scaleA, B1, scaleB1, gamma, B2, scaleB2, eps, residual=residual
        )
        return GemmRmsNormGemmOutput(out=out, scaleA=scaleA2, residual=residual_out)

    raise ValueError(
        f"unsupported dtype combo: A={A.dtype}, B1={B1.dtype}, B2={B2.dtype}"
    )

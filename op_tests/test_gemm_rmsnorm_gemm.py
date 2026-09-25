# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
try:
    import pytest
except ImportError:  # allow running as a plain script without pytest
    pytest = None
import numpy as np
import torch

import aiter
from aiter.ops.gemm_rmsnorm_gemm import (
    _e8m0,
    _unswizzle_gfx950_np,
    quantize_mxfp8_gfx950,
    quantize_mxfp8_weight_nhid,
)

CASES = [
    (1024, 64, 1024, 64),
    (512, 64, 1024, 64),
    (1024, 64, 512, 64),
    (2048, 64, 1024, 128),
]

_BF16_UNAVAIL_REASON = (
    "bf16 Chain: UNAVAILABLE with current hipblaslt "
    "(no PARTIAL_RMSNORM_STATS-only solution; algoCount==0)"
)


def _ref(A, W1, gamma, W2, eps, residual=None):
    h1 = A.float() @ W1.float()
    if residual is not None:
        h1 = h1 + residual.float()
    rstd = torch.rsqrt(h1.pow(2).mean(dim=-1, keepdim=True) + eps)  # [M,1]
    h2 = (h1 * gamma.float()).to(torch.bfloat16).float()  # producer stores h1*gamma in bf16
    return rstd * (h2 @ W2.float())


def _run_case(M, K1, Nhidden, Nout):
    torch.manual_seed(0)
    eps    = 1e-5
    A      = (torch.randn(M, K1, device="cuda") * 0.1).to(torch.bfloat16)
    W1     = (torch.randn(K1, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)
    gamma  = (torch.rand(Nhidden, device="cuda") + 0.5).to(torch.bfloat16)
    W2     = (torch.randn(Nhidden, Nout, device="cuda") * 0.1).to(torch.bfloat16)
    res_in = (torch.randn(M, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)

    try:
        out, residual_out = aiter.gemm_rmsnorm_gemm_bf16(
            A, W1, gamma, W2, eps, residual_in=res_in)
    except RuntimeError as e:
        if "algoCount==0" in str(e):
            print(f"  bf16 ({M},{K1},{Nhidden},{Nout}): UNAVAILABLE (algoCount==0)")
            if pytest is None:
                return "skip"
        raise
    ref = _ref(A, W1, gamma, W2, eps, residual=res_in)

    abs_err = (out.float() - ref).abs()
    tol = torch.maximum(torch.full_like(ref, 5e-2), 5e-2 * ref.abs())
    mism = (abs_err > tol).sum().item()
    assert mism == 0, f"mismatches={mism} max_abs={abs_err.max().item()}"

    H_ref = (A.float() @ W1.float() + res_in.float()).to(torch.bfloat16)
    res_diff = (residual_out.float() - H_ref.float()).abs().max().item()
    assert res_diff < 1e-2, f"bf16 residual_out mismatch max_abs={res_diff}"
    return "pass"


def _build_w2deq(B2_fp8: torch.Tensor, scaleB2: torch.Tensor, nOut: int, nHid: int) -> torch.Tensor:
    """Dequantize B2 using un-swizzled consumer-B scale; returns [nOut, nHid] float32."""
    block_size  = 32
    nHid_tiles  = (nHid + block_size - 1) // block_size
    padded_rows = ((nOut + 31) // 32) * 32
    padded_cols = ((nHid_tiles + 7) // 8) * 8
    scale_plain = _unswizzle_gfx950_np(scaleB2.cpu().numpy(), padded_rows, padded_cols)

    B2_f32 = B2_fp8.float().cpu().numpy()
    W2deq  = np.zeros((nOut, nHid), dtype=np.float32)
    for no in range(nOut):
        for nhblock in range(nHid_tiles):
            sb     = int(scale_plain[no, nhblock])
            dq     = 2.0 ** (sb - 127) if sb != 0 else 0.0
            nh_s   = nhblock * block_size
            nh_e   = min(nh_s + block_size, nHid)
            W2deq[no, nh_s:nh_e] = B2_f32[no, nh_s:nh_e] * dq
    return torch.from_numpy(W2deq)


def _dequant_mxfp8_gfx950(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize fp8 [nRows,nCols] with GFX950-swizzled UE8M0 block-32 scale (blocks along cols)."""
    nRows, nCols = x_fp8.shape
    block_size  = 32
    nTiles      = (nCols + block_size - 1) // block_size
    padded_rows = ((nRows + 31) // 32) * 32
    padded_cols = ((nTiles + 7) // 8) * 8
    scale_plain = _unswizzle_gfx950_np(scale.cpu().numpy(), padded_rows, padded_cols)
    x_f32 = x_fp8.float().cpu().numpy()
    out   = np.zeros((nRows, nCols), dtype=np.float32)
    for r in range(nRows):
        for tb in range(nTiles):
            sb = int(scale_plain[r, tb])
            dq = 2.0 ** (sb - 127) if sb != 0 else 0.0
            cs = tb * block_size
            ce = min(cs + block_size, nCols)
            out[r, cs:ce] = x_f32[r, cs:ce] * dq
    return torch.from_numpy(out)


def _mxfp8_quant_dequant_along_nhid(x: torch.Tensor) -> torch.Tensor:
    """Simulate producer fp8 D1 quantization: per-token, blocks of 32 along nHid.

    Mirrors the C++ e8m0QuantMult scheme applied to hg = h1*gamma.
    Returns a float32 tensor of the same shape after quantize-then-dequantize.
    A Python loop over nBlocks (e.g. 64) is acceptable for test shapes.
    """
    mTok, nHid = x.shape
    block_size = 32
    nBlocks = (nHid + block_size - 1) // block_size
    x_f32 = x.float()
    out = torch.zeros_like(x_f32)

    for bt in range(nBlocks):
        col_s = bt * block_size
        col_e = min(col_s + block_size, nHid)
        blk = x_f32[:, col_s:col_e].contiguous()  # [mTok, blk_w]

        amax = blk.abs().amax(dim=-1)  # [mTok] float32

        # Vectorized _e8m0 over all tokens in the block.
        scale_f = (amax / 448.0).contiguous()
        bits = scale_f.view(torch.int32)
        exp_byte = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        ceil_adj = (mant != 0).to(torch.int32)
        sb = (exp_byte + ceil_adj).clamp(max=254)
        q_exp = (254 - sb).clamp(min=1, max=254)
        qmult = q_exp.bitwise_left_shift(23).view(torch.float32)

        zero_mask = (amax == 0.0)
        qmult = qmult.masked_fill(zero_mask, 0.0)

        # Quantize to fp8, then dequantize back to float32.
        blk_q = (blk * qmult.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
        dq_scale = torch.pow(2.0, sb.float() - 127.0).masked_fill(zero_mask, 0.0)
        out[:, col_s:col_e] = blk_q * dq_scale.unsqueeze(-1)

    return out


MXFP8_CASES = [
    (256, 128, 2048, 64),       # small smoke
    (1, 1024, 4096, 4096),      # decode M=1, production hidden size
    (2048, 1024, 4096, 4096),   # prefill ISL=2048, production hidden size
]


def _run_fp8in_case(mTok, K1, nHid, nOut):
    torch.manual_seed(2)
    eps     = 1e-5
    a       = (torch.randn(mTok, K1)   * 0.1).to(torch.bfloat16).cuda()
    b1      = (torch.randn(nHid, K1)   * 0.1).to(torch.bfloat16).cuda()
    gamma   = (torch.rand(nHid)  + 0.5).to(torch.bfloat16).cuda()
    w2      = (torch.randn(nOut, nHid) * 0.1).to(torch.bfloat16).cuda()
    residual = (torch.randn(mTok, nHid) * 0.1).to(torch.bfloat16).cuda()

    a_fp8,  scaleA  = quantize_mxfp8_gfx950(a)
    b1_fp8, scaleB1 = quantize_mxfp8_gfx950(b1)
    b2_fp8, scaleB2 = quantize_mxfp8_weight_nhid(w2)

    out, scaleA2, residual_out = aiter.gemm_rmsnorm_gemm_mxfp8(
        a_fp8, scaleA, b1_fp8, scaleB1, gamma, b2_fp8, scaleB2, eps, residual=residual)

    assert residual_out.shape == (mTok, nHid), f"residual shape={residual_out.shape}"
    assert residual_out.dtype == torch.bfloat16, f"residual dtype={residual_out.dtype}"

    # Device-accurate reference: dequantize the exact fp8 operands the device sees.
    a_deq  = _dequant_mxfp8_gfx950(a_fp8, scaleA)        # [mTok, K1]
    b1_deq = _dequant_mxfp8_gfx950(b1_fp8, scaleB1)      # [nHid, K1]
    h1     = a_deq @ b1_deq.T + residual.float().cpu()    # [mTok, nHid] with residual
    rstd   = torch.rsqrt(h1.pow(2).mean(dim=-1, keepdim=True) + eps)
    hg     = h1 * gamma.float().cpu()
    hg_deq = _mxfp8_quant_dequant_along_nhid(hg)          # model producer D1 fp8
    w2deq  = _build_w2deq(b2_fp8, scaleB2, nOut, nHid)
    ref    = rstd * (hg_deq @ w2deq.T)                    # [mTok, nOut]

    out_cpu = out.float().cpu()
    rel_l2  = ((out_cpu - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    abs_err = (out_cpu - ref).abs()
    max_abs = abs_err.max().item()

    print(f"  fp8in shape ({mTok},{K1},{nHid},{nOut}): "
          f"rel_l2={rel_l2:.3e}, max_abs={max_abs:.3e}")

    assert scaleA2.any(), "scaleA2 is all zeros"
    # fp8 inputs add quantization error on top of the bf16-input chain; the
    # aggregate relative L2 error is the robust correctness signal.
    assert rel_l2 < 0.1, f"fp8in rel_l2={rel_l2:.3e} exceeds 0.1"

    # Residual numerical check (skip for mTok not a multiple of 128 due to library limitation).
    if mTok % 128 != 0:
        print(f"  fp8in residual ({mTok},{K1},{nHid},{nOut}): "
              f"skipping numerical check (mTok={mTok} not multiple of 128)")
        return

    ref_residual = a_deq @ b1_deq.T + residual.float().cpu()  # H = A @ W1 + residual_in
    res_f32  = residual_out.float().cpu()
    abs_err_r = (res_f32 - ref_residual).abs()
    max_abs_r = abs_err_r.max().item()
    nonzero   = ref_residual.abs() > 1e-3
    max_rel_r = (abs_err_r[nonzero] / ref_residual[nonzero].abs()).max().item() if nonzero.any() else 0.0

    abs_tol  = 2e-2
    rel_tol  = 0.10
    mismatch = (abs_err_r > torch.maximum(
        torch.full_like(ref_residual, abs_tol), rel_tol * ref_residual.abs())).sum().item()
    total    = mTok * nHid
    allowed  = max(8, int(0.0005 * mTok * nHid))

    print(f"  fp8in residual ({mTok},{K1},{nHid},{nOut}): "
          f"mismatch={mismatch}/{total} (allowed={allowed}), "
          f"max_abs={max_abs_r:.3e}, max_rel={max_rel_r:.3e}")

    assert residual_out.abs().sum() > 0, "fp8in residual is all zeros"
    assert mismatch <= allowed, (
        f"fp8in residual mismatch={mismatch}/{total} exceeds {allowed}")


if pytest is not None:
    @pytest.mark.parametrize("M,K1,Nhidden,Nout", CASES)
    def test_gemm_rmsnorm_gemm_bf16(M, K1, Nhidden, Nout):
        _run_case(M, K1, Nhidden, Nout)

    @pytest.mark.parametrize("mTok,K1,nHid,nOut", MXFP8_CASES)
    def test_gemm_rmsnorm_gemm_mxfp8(mTok, K1, nHid, nOut):
        _run_fp8in_case(mTok, K1, nHid, nOut)

    @pytest.mark.parametrize("M,K1,Nhidden,Nout", CASES)
    def test_unified_bf16(M, K1, Nhidden, Nout):
        torch.manual_seed(0)
        eps    = 1e-5
        A      = (torch.randn(M, K1, device="cuda") * 0.1).to(torch.bfloat16)
        # B1=[nHid,K1], B2=[nOut,nHid] — N,K layout expected by the unified dispatcher.
        B1_nk  = (torch.randn(Nhidden, K1, device="cuda") * 0.1).to(torch.bfloat16)
        gamma  = (torch.rand(Nhidden, device="cuda") + 0.5).to(torch.bfloat16)
        B2_nk  = (torch.randn(Nout, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)
        res_in = (torch.randn(M, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)

        res = aiter.gemm_rmsnorm_gemm(A, B1_nk, gamma, B2_nk, eps=eps, residual=res_in)
        assert res.scaleA is None, "bf16 chain must return scaleA=None"
        assert res.residual is not None, "bf16 chain must return residual"
        assert res.out.shape == (M, Nout), f"unexpected output shape {res.out.shape}"

        # Reference uses W1=[K1,nHid], W2=[nHid,nOut] (math layout).
        ref = _ref(A, B1_nk.T, gamma, B2_nk.T, eps, residual=res_in)
        abs_err = (res.out.float() - ref).abs()
        tol = torch.maximum(torch.full_like(ref, 5e-2), 5e-2 * ref.abs())
        mism = (abs_err > tol).sum().item()
        assert mism == 0, f"mismatches={mism} max_abs={abs_err.max().item()}"

        H_ref = (A.float() @ B1_nk.T.float() + res_in.float()).to(torch.bfloat16)
        res_diff = (res.residual.float() - H_ref.float()).abs().max().item()
        assert res_diff == 0.0, f"bf16 unified residual_out mismatch max_abs={res_diff}"

    @pytest.mark.parametrize("mTok,K1,nHid,nOut", MXFP8_CASES)
    def test_unified_fp8in(mTok, K1, nHid, nOut):
        torch.manual_seed(2)
        eps     = 1e-5
        a       = (torch.randn(mTok, K1)   * 0.1).to(torch.bfloat16).cuda()
        b1      = (torch.randn(nHid, K1)   * 0.1).to(torch.bfloat16).cuda()
        gamma   = (torch.rand(nHid)  + 0.5).to(torch.bfloat16).cuda()
        w2      = (torch.randn(nOut, nHid) * 0.1).to(torch.bfloat16).cuda()
        residual = (torch.randn(mTok, nHid) * 0.1).to(torch.bfloat16).cuda()

        a_fp8,  scaleA  = quantize_mxfp8_gfx950(a)
        b1_fp8, scaleB1 = quantize_mxfp8_gfx950(b1)
        b2_fp8, scaleB2 = quantize_mxfp8_weight_nhid(w2)

        # Direct call for reference.
        ref_out, ref_scaleA, _ = aiter.gemm_rmsnorm_gemm_mxfp8(
            a_fp8, scaleA, b1_fp8, scaleB1, gamma, b2_fp8, scaleB2, eps, residual=residual)

        # Unified dispatcher call.
        res = aiter.gemm_rmsnorm_gemm(
            a_fp8, b1_fp8, gamma, b2_fp8,
            scaleA=scaleA, scaleB1=scaleB1, scaleB2=scaleB2, eps=eps, residual=residual)
        assert res.out.shape == ref_out.shape, f"shape mismatch {res.out.shape} vs {ref_out.shape}"
        assert res.scaleA is not None, "fp8-in chain must return a scaleA"
        assert res.residual is not None, "fp8-in chain must return residual"
        assert torch.equal(res.out, ref_out), "unified fp8-in output differs from direct call"

    def test_unified_dispatch_errors():
        bf16 = torch.bfloat16
        fp8  = torch.float8_e4m3fn
        dev  = "cuda"
        A_bf  = torch.zeros(4, 8, dtype=bf16, device=dev)
        B1_bf = torch.zeros(16, 8, dtype=bf16, device=dev)
        g     = torch.ones(16, dtype=bf16, device=dev)
        B2_bf = torch.zeros(4, 16, dtype=bf16, device=dev)
        B2_fp = torch.zeros(4, 16, dtype=fp8, device=dev)
        sc    = torch.zeros(4, dtype=torch.uint8, device=dev)
        residual = torch.zeros(4, 16, dtype=bf16, device=dev)

        import pytest as _pytest
        # bf16 chain rejects any non-None scale.
        with _pytest.raises(ValueError, match="no scales"):
            aiter.gemm_rmsnorm_gemm(A_bf, B1_bf, g, B2_bf, scaleB2=sc, residual=residual)
        # bf16-A / fp8-B2 combo is not a supported chain.
        with _pytest.raises(ValueError, match="unsupported dtype"):
            aiter.gemm_rmsnorm_gemm(A_bf, B1_bf, g, B2_fp, scaleB2=sc, residual=residual)
        # fp8-in chain requires all three scales.
        A_fp  = torch.zeros(4, 8, dtype=fp8, device=dev)
        B1_fp = torch.zeros(16, 8, dtype=fp8, device=dev)
        with _pytest.raises(ValueError, match="requires scaleA"):
            aiter.gemm_rmsnorm_gemm(A_fp, B1_fp, g, B2_fp, residual=residual)
        # Unsupported dtype combo.
        B2_f32 = torch.zeros(4, 16, dtype=torch.float32, device=dev)
        with _pytest.raises(ValueError, match="unsupported dtype"):
            aiter.gemm_rmsnorm_gemm(A_bf, B1_bf, g, B2_f32, residual=residual)

if __name__ == "__main__":
    bf16_results = []
    for c in CASES:
        result = _run_case(*c)
        bf16_results.append(result)

    all_bf16_skipped = all(r == "skip" for r in bf16_results)
    any_bf16_failed  = any(r not in ("pass", "skip") for r in bf16_results)
    any_bf16_passed  = any(r == "pass" for r in bf16_results)

    if all_bf16_skipped:
        print(f"\n*** {_BF16_UNAVAIL_REASON} ***\n")
    elif any_bf16_passed:
        print(f"bf16 Chain: PASS ({bf16_results.count('pass')}/{len(CASES)} cases)")

    fp8in_passed = True
    for c in MXFP8_CASES:
        try:
            _run_fp8in_case(*c)
        except AssertionError as exc:
            print(f"  FP8IN FAIL: {exc}")
            fp8in_passed = False

    if fp8in_passed:
        print("FP8IN CHAIN PASS")
    else:
        print("FP8IN CHAIN FAIL")

    if any_bf16_failed:
        print("bf16 Chain: FAIL")

    # Unified dispatcher — bf16 chain.
    unified_bf16_passed = True
    for M, K1, Nhidden, Nout in CASES:
        torch.manual_seed(0)
        eps    = 1e-5
        A      = (torch.randn(M, K1, device="cuda") * 0.1).to(torch.bfloat16)
        B1_nk  = (torch.randn(Nhidden, K1, device="cuda") * 0.1).to(torch.bfloat16)
        gamma  = (torch.rand(Nhidden, device="cuda") + 0.5).to(torch.bfloat16)
        B2_nk  = (torch.randn(Nout, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)
        res_in = (torch.randn(M, Nhidden, device="cuda") * 0.1).to(torch.bfloat16)
        try:
            res = aiter.gemm_rmsnorm_gemm(A, B1_nk, gamma, B2_nk, eps=eps, residual=res_in)
            assert res.scaleA is None and res.residual is not None
            ref = _ref(A, B1_nk.T, gamma, B2_nk.T, eps, residual=res_in)
            abs_err = (res.out.float() - ref).abs()
            tol = torch.maximum(torch.full_like(ref, 5e-2), 5e-2 * ref.abs())
            mism = (abs_err > tol).sum().item()
            assert mism == 0, f"mismatches={mism}"
            H_ref = (A.float() @ B1_nk.T.float() + res_in.float()).to(torch.bfloat16)
            res_diff = (res.residual.float() - H_ref.float()).abs().max().item()
            assert res_diff < 1e-2, f"bf16 residual_out mismatch max_abs={res_diff}"
        except (RuntimeError, AssertionError) as exc:
            print(f"  UNIFIED BF16 FAIL ({M},{K1},{Nhidden},{Nout}): {exc}")
            unified_bf16_passed = False
    if unified_bf16_passed:
        print("UNIFIED BF16 PASS")
    else:
        print("UNIFIED BF16 FAIL")

    # Unified dispatcher — fp8-in chain.
    unified_fp8in_passed = True
    for mTok, K1, nHid, nOut in MXFP8_CASES:
        torch.manual_seed(2)
        eps      = 1e-5
        a        = (torch.randn(mTok, K1)   * 0.1).to(torch.bfloat16).cuda()
        b1       = (torch.randn(nHid, K1)   * 0.1).to(torch.bfloat16).cuda()
        gamma    = (torch.rand(nHid)  + 0.5).to(torch.bfloat16).cuda()
        w2       = (torch.randn(nOut, nHid) * 0.1).to(torch.bfloat16).cuda()
        residual = (torch.randn(mTok, nHid) * 0.1).to(torch.bfloat16).cuda()
        a_fp8,  scaleA  = quantize_mxfp8_gfx950(a)
        b1_fp8, scaleB1 = quantize_mxfp8_gfx950(b1)
        b2_fp8, scaleB2 = quantize_mxfp8_weight_nhid(w2)
        try:
            ref_out, _, _ = aiter.gemm_rmsnorm_gemm_mxfp8(
                a_fp8, scaleA, b1_fp8, scaleB1, gamma, b2_fp8, scaleB2, eps, residual=residual)
            res = aiter.gemm_rmsnorm_gemm(
                a_fp8, b1_fp8, gamma, b2_fp8,
                scaleA=scaleA, scaleB1=scaleB1, scaleB2=scaleB2, eps=eps, residual=residual)
            assert torch.equal(res.out, ref_out), "unified fp8-in output mismatch"
        except AssertionError as exc:
            print(f"  UNIFIED FP8IN FAIL ({mTok},{K1},{nHid},{nOut}): {exc}")
            unified_fp8in_passed = False
    if unified_fp8in_passed:
        print("UNIFIED FP8IN PASS")
    else:
        print("UNIFIED FP8IN FAIL")

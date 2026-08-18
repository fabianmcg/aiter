<div align="center">
<img src="docs/assets/aiter_logo.png" alt="AITER" width="400">
<br><br>

[![CI](https://github.com/ROCm/aiter/actions/workflows/aiter-test.yaml/badge.svg)](https://github.com/ROCm/aiter/actions/workflows/aiter-test.yaml)
[![Release](https://img.shields.io/github/v/release/ROCm/aiter)](https://github.com/ROCm/aiter/releases)
[![Docs](https://img.shields.io/badge/Docs-rocm.github.io%2Faiter-blue)](https://rocm.github.io/aiter)
[![Last Commit](https://img.shields.io/github/last-commit/ROCm/aiter)](https://github.com/ROCm/aiter/commits)

</div>

--------------------------------------------------------------------------------

**AITER** (AI Tensor Engine for ROCm) is AMD's high-performance AI operator library, providing optimized GPU kernels for inference and training workloads on ROCm. It serves as a unified collection of production-ready operators that framework developers can integrate directly into their stacks.

### Key Features

- **C++ and Python APIs** — use operators from either level
- **Multiple kernel backends** — Triton, Composable Kernel (CK), and hand-tuned ASM
- **Inference and training** — not just serving kernels, but also training and GEMM+communication fused kernels
- **Framework-agnostic** — integrate into vLLM, SGLang, or any custom framework

## News

- **[2026/07]** [Kimi-K3 support](https://github.com/ROCm/aiter/pull/4397) — FlyDSL SiTUv2 fused-MoE kernels, strided grouped-topk router, and tuned GEMM/fused-MoE configs (BF16, A8W4, FP4) for Kimi-K3
- **[2026/04]** [AITER v0.1.12.post1 Released](https://github.com/ROCm/aiter/releases/tag/v0.1.12.post1) — patch on v0.1.12 with GEMM and scale masking accuracy fixes; v0.1.12 highlights include blockwise sparse Sage Attention, fused gated RMSNorm+group quantization, etc., plus MI355X tuned configs for Kimi-K2.5 and DeepSeek-V3
- **[2026/02]** [JAX-AITER: Bringing AMD's Optimized AI Kernels to JAX on ROCm](https://rocm.blogs.amd.com/software-tools-optimization/jax-aiter/README.html)
- **[2026/02]** [Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
- **[2026/01]** [Character.ai: 2x Production Inference Performance on AMD Instinct GPUs](https://blog.character.ai/technical-deep-dive-how-digitalocean-and-amd-delivered-a-2x-production-inference-performance-increase-for-character-ai/)
- **[2026/01]** [ROCm Becomes a First-Class Platform in the vLLM Ecosystem](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)
- **[2025]** [Accelerated LLM Inference with vLLM 0.9.x and ROCm](https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html)
- **[2025]** [Accelerate DeepSeek-R1 Inference: Integrate AITER into SGLang](https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html)
- **[2025/08]** [AITER-Enabled MLA Layer Inference on AMD Instinct MI300X](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html)
- **[2025/08]** [Tutorial: MLA Decoding Kernel of the AITER Library to Accelerate LLM Inference](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- **[2025/03]** [Accelerating DeepSeek Inference with AMD MI300 — Microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/accelerating-deepseek-inference-with-amd-mi300-a-collaborative-breakthrough/4407673)
- **[2025/03]** [AITER: AI Tensor Engine For ROCm — Launch Announcement](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)

## Ecosystem

AITER is the **default kernel backend for LLM inference on AMD GPUs**, integrated into the major serving frameworks and powering production workloads at scale.

### Framework Integration

| Framework | Integration | Status | Operators Used |
|---|---|---|---|
| [**vLLM**](https://github.com/vllm-project/vllm) | Default attention backend on ROCm | Production | MHA, MLA, Paged Attention, Fused MoE, GEMM, RMSNorm, RoPE+KVCache |
| [**SGLang**](https://github.com/sgl-project/sglang) | Default on ROCm Docker | Production | Attention, Fused MoE, Block-scale GEMM, All-reduce, RMSNorm |
| [**ATOM**](https://github.com/ROCm/ATOM) | Built natively on AITER | Active development | All AITER operators (attention, MoE, sampling, communication) |
| [**JAX**](https://github.com/ROCm/jax-aiter) | XLA FFI bridge, no PyTorch dependency | Experimental | MHA/FMHA, RMSNorm, BF16 GEMM |
| Various customer proprietary inference engines | Kernel-level integration | Production | Attention, MoE, GEMM, quantization |

### Performance Highlights

| Operator | Speedup |
|---|---|
| MLA decode kernel | up to **17x** |
| MHA prefill kernel | up to **14x** |
| Block-scaled Fused MoE | up to **3x** |
| Block-scaled GEMM | up to **2x** |
| DeepSeek-R1 e2e (SGLang) | 6,484 → **13,704** tok/s (2.1x) |
| JAX-AITER attention (MI350) | **4.39x** median |

> For detailed benchmarks, see the [ATOM Benchmark Dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

### Supported Hardware

| GPU | Architecture | Status |
|---|---|---|
| AMD Instinct MI300X | gfx942 (CDNA3) | Fully supported |
| AMD Instinct MI325X | gfx942 (CDNA3) | Fully supported |
| AMD Instinct MI350 | gfx950 (CDNA4) | Supported |
| AMD Instinct MI355X | gfx950 (CDNA4) | Supported |
| AMD Pro W7900 | gfx1100 (RDNA3) | Experimental<sup>1</sup> |
| AMD AI Max and Max Pro 400/300 Series | gfx1151 (RDNA3.5) | Experimental<sup>1</sup> |
| AMD Radeon AI PRO R9700 | gfx1201 (RDNA4) | Experimental<sup>1</sup> |

<sup>1</sup> On RDNA, Triton and most FlyDSL kernels run, as do most HIP kernels (norm, RoPE, quant, activation, plus some GEMM/attention). Most CK and ASM kernels are CDNA-only.

## Operators

AITER provides optimized kernels for attention, MoE, GEMM, normalization, quantization, communication, and more. Each operator has unit tests under [`op_tests/`](op_tests/) that you can run directly:

```bash
# Example: run a single operator test
python3 op_tests/test_mha.py
python3 op_tests/test_mla.py
python3 op_tests/test_moe.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_rmsnorm2d.py

# See all available operator tests
ls op_tests/test_*.py
```

## Installation

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you happen to forget the `--recursive` during `clone`, you can use the following command after `cd aiter`
```bash
git submodule sync && git submodule update --init --recursive
```

### FlyDSL

AITER uses [FlyDSL](https://github.com/ROCm/FlyDSL)-based kernels across a range of operators (e.g., GEMM and MoE). FlyDSL is a required dependency and is installed automatically when you run `python3 setup.py develop`.

To install it manually:

```bash
pip install -r requirements.txt
```

### Triton

AITER includes Triton-based operators that require triton from AMD PyPI, with the correct version selected based on your ROCm installation.

If you install with `python3 setup.py develop`, triton is installed automatically. To skip this and keep your existing triton, set:

```bash
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
```

If you use `pip install -e .`, run the install script manually:

```bash
./.github/scripts/install_triton.sh
```

### Opus — Lightweight C++ Template for Kernel Development

[Opus](csrc/include/opus/) is a single-header C++ template library (`opus.hpp`) for writing HIP kernels on AMD GPUs — vectorized load/store, layout abstractions, and MFMA wrappers with a strong focus on **build time optimization** (up to 61x faster than standard torch extension builds). See the [Opus README](csrc/include/opus/README.md) and [`op_tests/opus/`](op_tests/opus/) for details.

### Triton-based Communication (Iris)

AITER supports GPU-initiated communication using the [Iris library](https://github.com/ROCm/iris). This enables high-performance Triton-based communication primitives like reduce-scatter and all-gather.

```bash
pip install -e .
pip install -r requirements-triton-comms.txt
```

For more details, see [docs/triton_comms.md](docs/triton_comms.md).

### Fused GEMM + RMSNorm + GEMM (gfx950 only)

The fused-epilogue GEMM chain (`gemm_rmsnorm_gemm_*`) requires a custom build of hipblaslt that exposes the epilogue API. This is a **gfx950-only** feature (MI350 / MI355X); it is a no-op on gfx942 (MI300X / MI325X).

**Requirement:** set `AITER_HIPBLASLT_PATH` to the absolute path of your custom `libhipblaslt.so` before building and before running. The same variable drives both the JIT compile flags (`-I`, `-L`, `-DCUSTOM_HIPBLASLT_PATH`) and the runtime `dlopen`. Without it, the module still imports but the first call raises a build error.

```bash
# Point at the .so file — e.g. from a local rocm-libraries build:
export AITER_HIPBLASLT_PATH=/path/to/hipblaslt-install/lib/libhipblaslt.so.1

# Then build (or rebuild if you already built without the variable):
AITER_REBUILD=1 python3 setup.py develop
```

The directory layout expected from `AITER_HIPBLASLT_PATH=/prefix/lib/libhipblaslt.so.1`:

| Component | Derived path |
|---|---|
| Include headers | `$AITER_HIPBLASLT_PATH/../../include` (i.e. `/prefix/include`) |
| Link library | `-L/prefix/lib -lhipblaslt` |
| Runtime rpath | `/prefix/lib` |

If you do not have a custom hipblaslt build, leave `AITER_HIPBLASLT_PATH` unset — the module is imported but gated by a `NotImplementedError` at the Python entry points.

## Build Configuration Reference

All environment variables that influence the build or runtime behavior:

### Build-time variables (set before `python3 setup.py develop`)

| Variable | Default | Description |
|---|---|---|
| `BUILD_TARGET` | `auto` | GPU target override (e.g. `gfx942`, `gfx950`). `auto` detects the installed GPU. |
| `PREBUILD_KERNELS` | `0` | Set to `1` to AOT-compile all kernels during install instead of lazily at first use. |
| `PRETUNE_MODULES` | `` | Comma-separated list of module names to pre-tune during install. |
| `ENABLE_CK` | `1` | Set to `0` to skip Composable Kernel (CK) compilation. Speeds up the build on machines where CK is not needed. |
| `AITER_TRITON_ONLY` | `0` | Set to `1` to build only the Triton/Python layer — skips all C++/HIP compilation. Useful on Windows or machines without a ROCm toolchain. |
| `AITER_USE_SYSTEM_TRITON` | `0` | Set to `1` to keep the already-installed triton rather than installing the AMD-pinned version. |
| `MAX_JOBS` | auto | Number of parallel compile jobs. Defaults to a heuristic based on CPU cores and free RAM. |
| `CK_DIR` | `3rdparty/composable_kernel` | Path to an external CK checkout to use instead of the bundled submodule. |
| `AITER_HIPBLASLT_PATH` | _(unset)_ | Absolute path to `libhipblaslt.so` for the fused GEMM + RMSNorm + GEMM epilogue feature (gfx950 only). See section above. |

### Runtime variables (set before running your workload)

| Variable | Default | Description |
|---|---|---|
| `AITER_REBUILD` | `0` | Set to `1` to force a clean JIT recompile of all modules on the next import, discarding any cached `.so`. |
| `AITER_HIPBLASLT_PATH` | _(unset)_ | Same as above — must also be set at runtime so the custom library can be `dlopen`-ed. |
| `AITER_LOG_MORE` | `0` | Set to `1` for verbose JIT build and dispatch logging. |
| `AITER_LOG_TUNED_CONFIG` | `0` | Set to `1` to log which tuned GEMM config is selected at each call site. |
| `AITER_DISABLE_KERNARG_PRELOAD` | `0` | Set to `1` to disable kernel-argument preloading (debugging aid). |
| `AITER_ENABLE_EXPERIMENTAL` | `0` | Set to `1` to enable experimental operators not yet in the stable API. |
| `AITER_CONFIG_GEMM_A8W8` | _(built-in CSV)_ | Path to a custom tuned-GEMM config CSV for A8W8 (overrides the bundled config). |
| `AITER_CONFIG_FMOE` | _(built-in CSV)_ | Path to a custom tuned fused-MoE config CSV. |

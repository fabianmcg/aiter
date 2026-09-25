// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <pybind11/pybind11.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(HIPBLASLT_HAS_GEMM_A2A_FUSION) || !HIPBLASLT_HAS_GEMM_A2A_FUSION
#  error "hipblaslt fused-epilogue API unavailable: set AITER_HIPBLASLT_PATH to a libhipblaslt.so built with HIPBLASLT_ENABLE_GEMM_A2A_FUSION=ON and rebuild"
#endif

namespace py = pybind11;

#define CHECK_HIPBLAS(expr)                                                             \
    do {                                                                                \
        hipblasStatus_t status_ = (expr);                                               \
        if (status_ != HIPBLAS_STATUS_SUCCESS)                                          \
            throw std::runtime_error(std::string("hipblaslt error at " #expr " = ") +  \
                                     std::to_string(static_cast<int>(status_)));        \
    } while (0)

// PyTorch loads system libhipblaslt.so.1 first; that version pre-dates the
// fused-epilogue API and the SONAME cache prevents the custom build from
// replacing it via RTLD_GLOBAL.  Opening the custom build by its ABSOLUTE PATH
// bypasses the SONAME cache and lets us retrieve each symbol via dlsym.
//
// AITER_HIPBLASLT_PATH (absolute path to the custom libhipblaslt.so.1) drives
// BOTH the compile-time flags (-I, -L, -rpath, -DCUSTOM_HIPBLASLT_PATH,
// injected by optCompilerConfig.json) AND the runtime dlopen path used by
// getHipblasltPath below.
#ifndef CUSTOM_HIPBLASLT_PATH
#  define CUSTOM_HIPBLASLT_PATH \
    "libhipblaslt.so.1"
#endif

static std::string getHipblasltPath()
{
    const char* env = std::getenv("AITER_HIPBLASLT_PATH");
    return env ? std::string(env) : std::string(CUSTOM_HIPBLASLT_PATH);
}

// ─── function pointer types ───────────────────────────────────────────────────
using pfnCreate =
    hipblasStatus_t (*)(hipblasLtHandle_t*);
using pfnMatrixLayoutCreate =
    hipblasStatus_t (*)(hipblasLtMatrixLayout_t*, hipDataType, uint64_t, uint64_t, int64_t);
using pfnMatrixLayoutDestroy =
    hipblasStatus_t (*)(const hipblasLtMatrixLayout_t);
using pfnMatmulDescCreate =
    hipblasStatus_t (*)(hipblasLtMatmulDesc_t*, hipblasComputeType_t, hipDataType);
using pfnMatmulDescSetAttribute =
    hipblasStatus_t (*)(hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t,
                        const void*, size_t);
using pfnMatmulDescDestroy =
    hipblasStatus_t (*)(const hipblasLtMatmulDesc_t);
using pfnMatmulPreferenceCreate =
    hipblasStatus_t (*)(hipblasLtMatmulPreference_t*);
using pfnMatmulPreferenceSetAttribute =
    hipblasStatus_t (*)(hipblasLtMatmulPreference_t,
                        hipblasLtMatmulPreferenceAttributes_t, const void*, size_t);
using pfnMatmulPreferenceDestroy =
    hipblasStatus_t (*)(const hipblasLtMatmulPreference_t);
using pfnMatmulAlgoGetHeuristic =
    hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
                        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                        hipblasLtMatmulPreference_t, int,
                        hipblasLtMatmulHeuristicResult_t*, int*);
using pfnMatmul =
    hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
                        const void*, const void*, hipblasLtMatrixLayout_t,
                        const void*, hipblasLtMatrixLayout_t,
                        const void*, const void*, hipblasLtMatrixLayout_t,
                        void*, hipblasLtMatrixLayout_t,
                        const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t);
using pfnFusedEpilogueCreate =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t*);
using pfnFusedEpilogueAdd =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t,
                        hipblasLtFuseableEpilogue_t);
using pfnFusedEpilogueSetAttribute =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t,
                        hipblasLtFusedEpilogueAttribute_t, const void*, size_t);
using pfnFusedEpilogueDestroy =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t);
using pfnFusedEpilogueRMSNormDescriptorCreate =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueRMSNormDescriptor_t*);
using pfnFusedEpilogueRMSNormDescriptorDestroy =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueRMSNormDescriptor_t);
using pfnFusedEpilogueRMSNormDescriptorSetBuffer =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueRMSNormDescriptor_t, void*, size_t);

// ─── vtable ──────────────────────────────────────────────────────────────────
// All calls go through this table so they always use the custom library,
// regardless of which hipblaslt version the SONAME cache returns.
struct HbltVtable {
    void*                                    lib;
    pfnCreate                                create;
    pfnMatrixLayoutCreate                    matrixLayoutCreate;
    pfnMatrixLayoutDestroy                   matrixLayoutDestroy;
    pfnMatmulDescCreate                      matmulDescCreate;
    pfnMatmulDescSetAttribute                matmulDescSetAttribute;
    pfnMatmulDescDestroy                     matmulDescDestroy;
    pfnMatmulPreferenceCreate                matmulPreferenceCreate;
    pfnMatmulPreferenceSetAttribute          matmulPreferenceSetAttribute;
    pfnMatmulPreferenceDestroy               matmulPreferenceDestroy;
    pfnMatmulAlgoGetHeuristic                matmulAlgoGetHeuristic;
    pfnMatmul                                matmul;
    pfnFusedEpilogueCreate                   fusedEpilogueCreate;
    pfnFusedEpilogueAdd                      fusedEpilogueAdd;
    pfnFusedEpilogueSetAttribute             fusedEpilogueSetAttribute;
    pfnFusedEpilogueDestroy                  fusedEpilogueDestroy;
    pfnFusedEpilogueRMSNormDescriptorCreate   rmsNormDescCreate;
    pfnFusedEpilogueRMSNormDescriptorDestroy  rmsNormDescDestroy;
    pfnFusedEpilogueRMSNormDescriptorSetBuffer rmsNormDescSetBuffer;

    template<typename F>
    F loadSym(const char* name) const
    {
        void* sym = dlsym(lib, name);
        if (!sym)
            throw std::runtime_error(std::string("hipblaslt symbol not found: ") + name);
        return reinterpret_cast<F>(sym);
    }

    HbltVtable()
    {
        lib = dlopen(getHipblasltPath().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (!lib)
            throw std::runtime_error(
                std::string("cannot dlopen custom hipblaslt: ") + dlerror());
        create                       = loadSym<pfnCreate>("hipblasLtCreate");
        matrixLayoutCreate           = loadSym<pfnMatrixLayoutCreate>("hipblasLtMatrixLayoutCreate");
        matrixLayoutDestroy          = loadSym<pfnMatrixLayoutDestroy>("hipblasLtMatrixLayoutDestroy");
        matmulDescCreate             = loadSym<pfnMatmulDescCreate>("hipblasLtMatmulDescCreate");
        matmulDescSetAttribute       = loadSym<pfnMatmulDescSetAttribute>("hipblasLtMatmulDescSetAttribute");
        matmulDescDestroy            = loadSym<pfnMatmulDescDestroy>("hipblasLtMatmulDescDestroy");
        matmulPreferenceCreate       = loadSym<pfnMatmulPreferenceCreate>("hipblasLtMatmulPreferenceCreate");
        matmulPreferenceSetAttribute = loadSym<pfnMatmulPreferenceSetAttribute>(
            "hipblasLtMatmulPreferenceSetAttribute");
        matmulPreferenceDestroy      = loadSym<pfnMatmulPreferenceDestroy>("hipblasLtMatmulPreferenceDestroy");
        matmulAlgoGetHeuristic       = loadSym<pfnMatmulAlgoGetHeuristic>("hipblasLtMatmulAlgoGetHeuristic");
        matmul                       = loadSym<pfnMatmul>("hipblasLtMatmul");
        fusedEpilogueCreate          = loadSym<pfnFusedEpilogueCreate>("hipblasLtFusedEpilogueCreate");
        fusedEpilogueAdd             = loadSym<pfnFusedEpilogueAdd>("hipblasLtFusedEpilogueAdd");
        fusedEpilogueSetAttribute    = loadSym<pfnFusedEpilogueSetAttribute>(
            "hipblasLtFusedEpilogueSetAttribute");
        fusedEpilogueDestroy         = loadSym<pfnFusedEpilogueDestroy>("hipblasLtFusedEpilogueDestroy");
        rmsNormDescCreate     = loadSym<pfnFusedEpilogueRMSNormDescriptorCreate>(
            "hipblasLtFusedEpilogueRMSNormDescriptorCreate");
        rmsNormDescDestroy    = loadSym<pfnFusedEpilogueRMSNormDescriptorDestroy>(
            "hipblasLtFusedEpilogueRMSNormDescriptorDestroy");
        rmsNormDescSetBuffer  = loadSym<pfnFusedEpilogueRMSNormDescriptorSetBuffer>(
            "hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer");
    }

    // Run custom-hipblaslt cleanup callbacks at process exit.
    ~HbltVtable()
    {
        if (lib)
            dlclose(lib);
    }
    HbltVtable(const HbltVtable&)            = delete;
    HbltVtable& operator=(const HbltVtable&) = delete;

    static const HbltVtable& get()
    {
        static HbltVtable instance;
        return instance;
    }
};

// RAII guards that destroy hipblaslt objects via the loaded vtable on scope exit.
struct MatrixLayoutGuard {
    hipblasLtMatrixLayout_t handle = nullptr;
    MatrixLayoutGuard()                                        = default;
    MatrixLayoutGuard(const MatrixLayoutGuard&)                = delete;
    MatrixLayoutGuard& operator=(const MatrixLayoutGuard&)     = delete;
    ~MatrixLayoutGuard() { if (handle) HbltVtable::get().matrixLayoutDestroy(handle); }
};
struct MatmulDescGuard {
    hipblasLtMatmulDesc_t handle = nullptr;
    MatmulDescGuard()                                      = default;
    MatmulDescGuard(const MatmulDescGuard&)                = delete;
    MatmulDescGuard& operator=(const MatmulDescGuard&)     = delete;
    ~MatmulDescGuard() { if (handle) HbltVtable::get().matmulDescDestroy(handle); }
};
struct MatmulPreferenceGuard {
    hipblasLtMatmulPreference_t handle = nullptr;
    MatmulPreferenceGuard()                                          = default;
    MatmulPreferenceGuard(const MatmulPreferenceGuard&)              = delete;
    MatmulPreferenceGuard& operator=(const MatmulPreferenceGuard&)   = delete;
    ~MatmulPreferenceGuard() { if (handle) HbltVtable::get().matmulPreferenceDestroy(handle); }
};
struct FusedEpilogueGuard {
    hipblasLtFusedEpilogueDescriptor_t handle = nullptr;
    FusedEpilogueGuard()                                       = default;
    FusedEpilogueGuard(const FusedEpilogueGuard&)              = delete;
    FusedEpilogueGuard& operator=(const FusedEpilogueGuard&)   = delete;
    ~FusedEpilogueGuard() { if (handle) HbltVtable::get().fusedEpilogueDestroy(handle); }
};
struct RmsNormStatsGuard {
    hipblasLtFusedEpilogueRMSNormDescriptor_t handle = nullptr;
    RmsNormStatsGuard()                                        = default;
    RmsNormStatsGuard(const RmsNormStatsGuard&)                = delete;
    RmsNormStatsGuard& operator=(const RmsNormStatsGuard&)     = delete;
    ~RmsNormStatsGuard() { if (handle) HbltVtable::get().rmsNormDescDestroy(handle); }
};

static hipblasLtHandle_t getHandle(int deviceIndex)
{
    static std::mutex mtx;
    static std::unordered_map<int, hipblasLtHandle_t> handles;
    std::lock_guard<std::mutex> lock(mtx);
    hipblasLtHandle_t& handle = handles[deviceIndex];
    if (!handle) {
        if (HbltVtable::get().create(&handle) != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error("failed to create hipblaslt handle");
    }
    return handle;
}

// Returns a pointer to a persistent, lazily-grown device workspace.
// The buffer lives for the lifetime of the process and is never freed, so
// hipblaslt can reuse it across calls without hitting the PyTorch allocator.
static std::pair<void*, size_t> getWorkspace(at::Device device)
{
    // 256 MB is the ceiling hipblaslt currently needs for fused-epilogue solutions.
    static constexpr size_t kWsSize = static_cast<size_t>(256) * 1024 * 1024;
    static std::mutex mtx;
    static std::unordered_map<int, torch::Tensor> workspaces;
    std::lock_guard<std::mutex> lock(mtx);
    const int deviceIndex = static_cast<int>(device.index());
    torch::Tensor& ws = workspaces[deviceIndex];
    if (!ws.defined())
        ws = torch::empty({static_cast<int64_t>(kWsSize)},
                          torch::TensorOptions().dtype(torch::kUInt8).device(device));
    return {ws.data_ptr(), kWsSize};
}

// Creates and configures the four matrix layouts and matmul descriptor for a TN fused bf16 gemm.
// Convention (post 8bc216ab): D.M = feature dim, D.N = token dim.  Caller passes weight as A
// ([k,M] col-major, lda=k) and activation as B ([k,N] col-major, ldb=k).
static void configureMatmulLayouts(const HbltVtable& v,
                                   int64_t m, int64_t n, int64_t k,
                                   hipblasLtFusedEpilogueDescriptor_t fused,
                                   MatrixLayoutGuard& gLayA, MatrixLayoutGuard& gLayB,
                                   MatrixLayoutGuard& gLayC, MatrixLayoutGuard& gLayD,
                                   MatmulDescGuard& gMm)
{
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayA.handle, HIP_R_16BF, k, m, k));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayB.handle, HIP_R_16BF, k, n, k));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayC.handle, HIP_R_16BF, m, n, m));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayD.handle, HIP_R_16BF, m, n, m));
    CHECK_HIPBLAS(v.matmulDescCreate(&gMm.handle, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(
        gMm.handle, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused)));
}

// Core TN fused matmul: op(A)=T, op(B)=N, alpha=1, beta=0, col-major layouts.
// A is weight [k,m] col-major (lda=k); B is activation [k,n] col-major (ldb=k).
// bf16 A/B/C/D.  Throws on hipblaslt errors or when no fused solution exists.
static void runTnFusedBf16(hipblasLtHandle_t                  handle,
                           int64_t                            m,
                           int64_t                            n,
                           int64_t                            k,
                           const void*                        dA,
                           const void*                        dB,
                           void*                              dC,
                           void*                              dD,
                           hipblasLtFusedEpilogueDescriptor_t fused,
                           void*                              dWs,
                           size_t                             wsSize,
                           hipStream_t                        stream)
{
    const HbltVtable& v = HbltVtable::get();
    MatrixLayoutGuard gLayA, gLayB, gLayC, gLayD;
    MatmulDescGuard gMm;
    configureMatmulLayouts(v, m, n, k, fused, gLayA, gLayB, gLayC, gLayD, gMm);

    MatmulPreferenceGuard gPref;
    CHECK_HIPBLAS(v.matmulPreferenceCreate(&gPref.handle));
    CHECK_HIPBLAS(v.matmulPreferenceSetAttribute(
        gPref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    constexpr int maxHeur = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heur(maxHeur);
    int algoCount = 0;
    CHECK_HIPBLAS(v.matmulAlgoGetHeuristic(
        handle, gMm.handle, gLayA.handle, gLayB.handle, gLayC.handle, gLayD.handle,
        gPref.handle, maxHeur, heur.data(), &algoCount));
    if (algoCount <= 0 || heur[0].state != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("no valid hipblaslt fused solution");

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_HIPBLAS(v.matmul(handle, gMm.handle, &alpha,
                           dA, gLayA.handle, dB, gLayB.handle,
                           &beta, dC, gLayC.handle, dD, gLayD.handle,
                           &heur[0].algo, dWs, wsSize, stream));
}

// Generalized configureMatmulLayouts with typed A/B/CD element types.
// Convention (post 8bc216ab): A is weight [k,m] col-major (lda=k), B is activation [k,n].
static void configureMatmulLayoutsTyped(const HbltVtable& v,
                                        hipDataType aType, hipDataType bType, hipDataType cdType,
                                        int64_t m, int64_t n, int64_t k,
                                        hipblasLtFusedEpilogueDescriptor_t fused,
                                        MatrixLayoutGuard& gLayA, MatrixLayoutGuard& gLayB,
                                        MatrixLayoutGuard& gLayC, MatrixLayoutGuard& gLayD,
                                        MatmulDescGuard& gMm)
{
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayA.handle, aType,  k, m, k));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayB.handle, bType,  k, n, k));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayC.handle, cdType, m, n, m));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayD.handle, cdType, m, n, m));
    CHECK_HIPBLAS(v.matmulDescCreate(&gMm.handle, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(
        gMm.handle, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused)));
}

// Sets BLK32_UE8M0 scale mode and pointer on the given matmul descriptor operand.
// No-op when scale is null.
static void setMxScaleAttrib(hipblasLtMatmulDesc_t mm,
                             hipblasLtMatmulDescAttributes_t modeAttr,
                             hipblasLtMatmulDescAttributes_t ptrAttr,
                             const void* scale)
{
    if (!scale)
        return;
    const HbltVtable& v = HbltVtable::get();
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
    CHECK_HIPBLAS(v.matmulDescSetAttribute(mm, modeAttr, &mode, sizeof(mode)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(mm, ptrAttr, &scale, sizeof(scale)));
}

// Generalized TN fused matmul with typed A/B/CD types and optional MX block scale pointers.
// A is weight, B is activation; caller swaps M/N so D.M=features, D.N=tokens.
static void runTnFused(hipblasLtHandle_t                  handle,
                       hipDataType                        aType,
                       hipDataType                        bType,
                       hipDataType                        cdType,
                       int64_t                            m,
                       int64_t                            n,
                       int64_t                            k,
                       const void*                        dA,
                       const void*                        dScaleA,
                       const void*                        dB,
                       const void*                        dScaleB,
                       void*                              dC,
                       void*                              dD,
                       hipblasLtFusedEpilogueDescriptor_t fused,
                       void*                              dWs,
                       size_t                             wsSize,
                       hipStream_t                        stream)
{
    const HbltVtable& v = HbltVtable::get();
    MatrixLayoutGuard gLayA, gLayB, gLayC, gLayD;
    MatmulDescGuard gMm;
    configureMatmulLayoutsTyped(v, aType, bType, cdType, m, n, k, fused,
                                gLayA, gLayB, gLayC, gLayD, gMm);
    setMxScaleAttrib(gMm.handle,
                     HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                     HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, dScaleA);
    setMxScaleAttrib(gMm.handle,
                     HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                     HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, dScaleB);
    MatmulPreferenceGuard gPref;
    CHECK_HIPBLAS(v.matmulPreferenceCreate(&gPref.handle));
    CHECK_HIPBLAS(v.matmulPreferenceSetAttribute(
        gPref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
    constexpr int maxHeur = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heur(maxHeur);
    int algoCount = 0;
    CHECK_HIPBLAS(v.matmulAlgoGetHeuristic(
        handle, gMm.handle, gLayA.handle, gLayB.handle, gLayC.handle, gLayD.handle,
        gPref.handle, maxHeur, heur.data(), &algoCount));
    if (algoCount <= 0 || heur[0].state != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("no valid hipblaslt fused solution");
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_HIPBLAS(v.matmul(handle, gMm.handle, &alpha,
                           dA, gLayA.handle, dB, gLayB.handle,
                           &beta, dC, gLayC.handle, dD, gLayD.handle,
                           &heur[0].algo, dWs, wsSize, stream));
}

// Builds the producer fused epilogue: RESIDUAL_ADD + PARTIAL_RMSNORM_STATS.
// residualInPtr: read-only residual-add source (RESIDUAL_POINTER).
// residualOutPtr: separate output buffer for H = A@W1+residual (RESIDUAL_OUTPUT_POINTER),
// written as bf16 by the PartialRMSStoreBf16D kernel path.
static void buildProducerEpilogue(const HbltVtable& v,
                                  void* gammaPtr,
                                  float epsF,
                                  hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                  void* residualInPtr,
                                  void* residualOutPtr,
                                  FusedEpilogueGuard& prod)
{
    CHECK_HIPBLAS(v.fusedEpilogueCreate(&prod.handle));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(prod.handle, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residualInPtr, sizeof(residualInPtr)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER, &residualOutPtr, sizeof(residualOutPtr)));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(prod.handle, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gammaPtr, sizeof(gammaPtr)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &epsF, sizeof(epsF)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)));
}

// Builds the consumer fused epilogue: RMSNORM_SCALE_APPLY reading from stats.
static void buildConsumerEpilogue(const HbltVtable& v,
                                  hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                  FusedEpilogueGuard& cons)
{
    CHECK_HIPBLAS(v.fusedEpilogueCreate(&cons.handle));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(cons.handle, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        cons.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)));
}

// Validates inputs and prepares contiguous operand tensors for the fused gemm-rmsnorm-gemm kernel.
static void validateAndPrepareOperands(torch::Tensor A, torch::Tensor W1,
                                       torch::Tensor gamma, torch::Tensor W2,
                                       int64_t& M, int64_t& K1, int64_t& Nhidden, int64_t& Nout,
                                       torch::Tensor& aC, torch::Tensor& b1,
                                       torch::Tensor& b2, torch::Tensor& gammaC)
{
    TORCH_CHECK(A.is_cuda() && W1.is_cuda() && W2.is_cuda() && gamma.is_cuda(),
                "all inputs must be cuda");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(W1.scalar_type() == at::kBFloat16, "W1 must be bf16");
    TORCH_CHECK(W2.scalar_type() == at::kBFloat16, "W2 must be bf16");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(A.dim() == 2 && W1.dim() == 2 && W2.dim() == 2 && gamma.dim() == 1,
                "bad dims");
    M       = A.size(0);
    K1      = A.size(1);
    Nhidden = W1.size(0);   // W1 is [Nhidden, K1]
    Nout    = W2.size(0);   // W2 is [Nout, Nhidden]
    TORCH_CHECK(W1.size(1) == K1,         "W1 cols must equal A cols");
    TORCH_CHECK(gamma.size(0) == Nhidden, "gamma length must equal Nhidden");
    TORCH_CHECK(W2.size(1) == Nhidden,    "W2 cols must equal Nhidden");
    // W1 and W2 are already in [N, K] layout; pass contiguous directly.
    aC     = A.contiguous();   // row-major [M, K1]
    b1     = W1.contiguous();  // row-major [Nhidden, K1]
    b2     = W2.contiguous();  // row-major [Nout, Nhidden]
    gammaC = gamma.contiguous();
}

// Builds PARTIAL_RMSNORM_STATS + REQUANT(MX-fp8) producer fused epilogue.
// residual_in is mandatory; always prepends RESIDUAL_ADD so the kernel computes
// H = A @ W1 + residualIn.  The PartialRMSStoreBf16D path writes H as bf16 into residualOut.
static void buildMxfp8ProducerEpilogue(const HbltVtable& v,
                                       void* gammaPtr, float epsF,
                                       hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                       void* dMxScale,
                                       void* residualOut,
                                       void* residualIn,
                                       FusedEpilogueGuard& prod)
{
    CHECK_HIPBLAS(v.fusedEpilogueCreate(&prod.handle));
    // Residual is mandatory; always add RESIDUAL_ADD stage.
    CHECK_HIPBLAS(v.fusedEpilogueAdd(prod.handle, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
        &residualIn, sizeof(residualIn)));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(prod.handle, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(prod.handle, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gammaPtr, sizeof(gammaPtr)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &epsF, sizeof(epsF)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)));
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)));
    int32_t bs = 32;
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)));
    hipDataType outType = HIP_R_8F_E4M3;
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        prod.handle, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER,
        &residualOut, sizeof(residualOut)));
}

// Returns the flat size of the MX scale byte buffer for a [mTok,nHid] tile.
static int64_t mxfp8ScaleBufferSize(int64_t mTok, int64_t nHid)
{
    const int64_t paddedRows = ((mTok + 31) / 32) * 32;
    const int64_t nTiles     = (nHid + 31) / 32;
    const int64_t paddedCols = ((nTiles + 7) / 8) * 8;
    return paddedRows * paddedCols;
}

// Holds the bf16 residual output buffer written by the RESIDUAL_ADD + PartialRMSStoreBf16D path.
struct ResidualBuffers {
    torch::Tensor residual;
    void* residualPtr = nullptr;
};

// Allocates the bf16 residual output buffer written by the RESIDUAL_ADD + PartialRMSStoreBf16D path.
static ResidualBuffers makeResidualBuffers(int64_t mTok, int64_t nHid,
                                           torch::TensorOptions optsBf16)
{
    ResidualBuffers bufs;
    bufs.residual    = torch::empty({mTok, nHid}, optsBf16);
    bufs.residualPtr = bufs.residual.data_ptr();
    return bufs;
}

// Validates and returns a contiguous data pointer for the mandatory bf16 residual input.
static void* prepareResidualIn(torch::Tensor residualIn, int64_t mTok, int64_t nHid,
                               torch::Tensor& residualInC)
{
    TORCH_CHECK(residualIn.is_cuda(), "residual_in must be cuda");
    TORCH_CHECK(residualIn.scalar_type() == at::kBFloat16, "residual_in must be bf16");
    TORCH_CHECK(residualIn.dim() == 2 && residualIn.size(0) == mTok && residualIn.size(1) == nHid,
                "residual_in must be [mTok, nHid]");
    residualInC = residualIn.contiguous();
    return residualInC.data_ptr();
}

// Validates and extracts dimensions for the fp8-input MXFP8 chain.
static void validateFp8InChainInputs(torch::Tensor A, torch::Tensor scaleA,
                                     torch::Tensor B1, torch::Tensor scaleB1,
                                     torch::Tensor gamma,
                                     torch::Tensor B2, torch::Tensor scaleB2,
                                     int64_t& mTok, int64_t& k1, int64_t& nHid, int64_t& nOut)
{
    TORCH_CHECK(A.is_cuda() && scaleA.is_cuda() && B1.is_cuda() && scaleB1.is_cuda() &&
                gamma.is_cuda() && B2.is_cuda() && scaleB2.is_cuda(), "all inputs must be cuda");
    TORCH_CHECK(A.scalar_type()      == at::kFloat8_e4m3fn, "A must be fp8 e4m3");
    TORCH_CHECK(scaleA.scalar_type() == at::kByte,          "scaleA must be uint8");
    TORCH_CHECK(B1.scalar_type()     == at::kFloat8_e4m3fn, "B1 must be fp8 e4m3");
    TORCH_CHECK(scaleB1.scalar_type()== at::kByte,          "scaleB1 must be uint8");
    TORCH_CHECK(gamma.scalar_type()  == at::kBFloat16,      "gamma must be bf16");
    TORCH_CHECK(B2.scalar_type()     == at::kFloat8_e4m3fn, "B2 must be fp8 e4m3");
    TORCH_CHECK(scaleB2.scalar_type()== at::kByte,          "scaleB2 must be uint8");
    TORCH_CHECK(A.dim() == 2 && B1.dim() == 2 && gamma.dim() == 1 && B2.dim() == 2, "bad dims");
    mTok = A.size(0);
    k1   = A.size(1);
    nHid = B1.size(0);
    nOut = B2.size(0);
    TORCH_CHECK(B1.size(1) == k1,      "B1 cols must equal A cols");
    TORCH_CHECK(gamma.size(0) == nHid, "gamma length must equal nHid");
    TORCH_CHECK(B2.size(1) == nHid,    "B2 cols must equal nHid");
}

// Runs fp8-input GEMM1 (fp8×fp8→fp8 + PARTIAL_RMSNORM_STATS + REQUANT) then
// consumer GEMM2 (fp8×fp8→bf16 + RMSNORM_SCALE_APPLY), sharing one stats descriptor.
static void runFp8InMxfp8Chain(const HbltVtable& v, hipblasLtHandle_t handle,
                               int64_t mTok, int64_t k1, int64_t nHid, int64_t nOut,
                               void* gammaPtr, float epsF,
                               const void* aCPtr, const void* saPtr,
                               const void* b1Ptr, const void* sb1Ptr,
                               const void* b2Ptr, const void* sb2Ptr,
                               void* d1Ptr, void* scaleAOutPtr, void* basePtr,
                               void* residualOutPtr, void* residualInPtr,
                               void* perRowScalePtr,
                               void* wsPtr, size_t wsSize, hipStream_t stream)
{
    RmsNormStatsGuard gStats;
    CHECK_HIPBLAS(v.rmsNormDescCreate(&gStats.handle));
    // Caller-owned FP32 rstd buffer: one value per token (D.N = mTok).
    CHECK_HIPBLAS(v.rmsNormDescSetBuffer(
        gStats.handle, perRowScalePtr, static_cast<size_t>(mTok) * sizeof(float)));

    FusedEpilogueGuard gProd;
    buildMxfp8ProducerEpilogue(v, gammaPtr, epsF, gStats.handle, scaleAOutPtr,
                               residualOutPtr, residualInPtr, gProd);
    // GEMM1: weight B1 as A, activation A as B; D.M=nHid (features), D.N=mTok (tokens).
    runTnFused(handle, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_8F_E4M3,
               nHid, mTok, k1, b1Ptr, sb1Ptr,
               aCPtr, saPtr,
               d1Ptr, d1Ptr, gProd.handle,
               wsPtr, wsSize, stream);

    FusedEpilogueGuard gCons;
    buildConsumerEpilogue(v, gStats.handle, gCons);
    // GEMM2: weight B2 as A, producer output D1 as B; D.M=nOut (features), D.N=mTok (tokens).
    runTnFused(handle, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF,
               nOut, mTok, nHid, b2Ptr, sb2Ptr,
               d1Ptr, scaleAOutPtr,
               basePtr, basePtr, gCons.handle,
               wsPtr, wsSize, stream);
}

// A:[M,K1] bf16, W1:[Nhidden,K1] bf16, gamma:[Nhidden] bf16, W2:[Nout,Nhidden] bf16.
// residual_in: [M,Nhidden] bf16, mandatory residual added to GEMM1 output before RMSNorm.
// Returns {out [M,Nout] bf16, residual_out [M,Nhidden] bf16} where
// residual_out = A @ W1 + residual_in stored as bf16.
std::vector<torch::Tensor> gemm_rmsnorm_gemm_bf16(torch::Tensor A,
                                                   torch::Tensor W1,
                                                   torch::Tensor gamma,
                                                   torch::Tensor W2,
                                                   double        eps,
                                                   torch::Tensor residualIn)
{
    int64_t M, K1, Nhidden, Nout;
    torch::Tensor aC, b1, b2, gammaC;
    validateAndPrepareOperands(A, W1, gamma, W2, M, K1, Nhidden, Nout, aC, b1, b2, gammaC);

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA deviceGuard{A.device()};
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    auto optsBf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A.device());

    // Producer output h2: row-major [M, Nhidden].
    torch::Tensor h2   = torch::empty({M, Nhidden}, optsBf16);
    // GEMM2 consumer (RMSNORM_SCALE_APPLY) transposes internally and writes a
    // contiguous row-major [M, Nout] output (transposeForScaleApply + UseScaleAlphaVec=2).
    torch::Tensor base = torch::empty({M, Nout}, optsBf16);

    torch::Tensor residualInC;
    prepareResidualIn(residualIn, M, Nhidden, residualInC);
    // Separate output buffer for H = A@W1+residual_in (written by RESIDUAL_OUTPUT_POINTER).
    torch::Tensor residualOut = torch::empty({M, Nhidden}, optsBf16);

    auto [wsPtr, wsSize] = getWorkspace(A.device());

    const HbltVtable& v      = HbltVtable::get();
    hipblasLtHandle_t  handle = getHandle(static_cast<int>(A.device().index()));

    RmsNormStatsGuard gStats;
    CHECK_HIPBLAS(v.rmsNormDescCreate(&gStats.handle));
    // Caller-owned FP32 rstd buffer: one value per token (D.N = M).
    torch::Tensor perRowScale = torch::empty({M}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(A.device()));
    CHECK_HIPBLAS(v.rmsNormDescSetBuffer(
        gStats.handle, perRowScale.data_ptr(), static_cast<size_t>(M) * sizeof(float)));

    FusedEpilogueGuard gProd;
    buildProducerEpilogue(v, gammaC.data_ptr(), static_cast<float>(eps), gStats.handle,
                          residualInC.data_ptr(), residualOut.data_ptr(), gProd);
    // GEMM1: weight b1 as A, activation aC as B; D.M=Nhidden (features), D.N=M (tokens).
    runTnFusedBf16(handle, Nhidden, M, K1, b1.data_ptr(), aC.data_ptr(),
                   h2.data_ptr(), h2.data_ptr(), gProd.handle, wsPtr, wsSize, stream);

    FusedEpilogueGuard gCons;
    buildConsumerEpilogue(v, gStats.handle, gCons);
    // GEMM2: weight b2 as A, h2 as B; D.M=Nout (features), D.N=M (tokens).
    runTnFusedBf16(handle, Nout, M, Nhidden, b2.data_ptr(), h2.data_ptr(),
                   base.data_ptr(), base.data_ptr(), gCons.handle, wsPtr, wsSize, stream);

    return {base, residualOut};
}

// A_fp8:[mTok,k1] fp8 e4m3, scaleA: UE8M0 GFX950-swizzled input-A scale,
// B1_fp8:[nHid,k1] fp8 e4m3, scaleB1: UE8M0 GFX950-swizzled input-B1 scale,
// gamma:[nHid] bf16, B2:[nOut,nHid] fp8 e4m3, scaleB2: UE8M0 GFX950-swizzled consumer-B2 scale.
// residual_in: [mTok,nHid] bf16, mandatory residual added before RMSNorm.
// Returns {out[mTok,nOut] bf16, scaleA2 UE8M0 bytes, residual_out [mTok,nHid] bf16}.
std::vector<torch::Tensor> gemm_rmsnorm_gemm_mxfp8(
    torch::Tensor A_fp8, torch::Tensor scaleA,
    torch::Tensor B1_fp8, torch::Tensor scaleB1,
    torch::Tensor gamma,
    torch::Tensor B2, torch::Tensor scaleB2,
    double eps, torch::Tensor residualIn)
{
    int64_t mTok, k1, nHid, nOut;
    validateFp8InChainInputs(A_fp8, scaleA, B1_fp8, scaleB1, gamma, B2, scaleB2,
                             mTok, k1, nHid, nOut);

    const int64_t scaleBufSz = mxfp8ScaleBufferSize(mTok, nHid);

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA deviceGuard{A_fp8.device()};
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    auto optsFp8  = torch::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_fp8.device());
    auto optsBf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp8.device());
    auto optsU8   = torch::TensorOptions().dtype(torch::kUInt8).device(A_fp8.device());

    torch::Tensor d1      = torch::empty({mTok, nHid}, optsFp8);
    torch::Tensor scaleA2 = torch::zeros({scaleBufSz}, optsU8);
    // GEMM2 consumer (RMSNORM_SCALE_APPLY) transposes internally and writes a
    // contiguous row-major [mTok, nOut] output (transposeForScaleApply + UseScaleAlphaVec=2).
    torch::Tensor base    = torch::empty({mTok, nOut}, optsBf16);

    torch::Tensor residualInC;
    void* residualInPtr = prepareResidualIn(residualIn, mTok, nHid, residualInC);
    ResidualBuffers resBufs = makeResidualBuffers(mTok, nHid, optsBf16);

    // Per-row FP32 scale buffer required by the updated hipblaslt before matmul.
    torch::Tensor perRowScale = torch::empty({mTok}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(A_fp8.device()));

    auto [wsPtr, wsSize]  = getWorkspace(A_fp8.device());

    torch::Tensor aC   = A_fp8.contiguous();
    torch::Tensor saC  = scaleA.contiguous();
    torch::Tensor b1C  = B1_fp8.contiguous();
    torch::Tensor sb1C = scaleB1.contiguous();
    torch::Tensor gcC  = gamma.contiguous();
    torch::Tensor b2C  = B2.contiguous();
    torch::Tensor sb2C = scaleB2.contiguous();

    runFp8InMxfp8Chain(HbltVtable::get(), getHandle(static_cast<int>(A_fp8.device().index())),
                       mTok, k1, nHid, nOut,
                       gcC.data_ptr(), static_cast<float>(eps),
                       aC.data_ptr(), saC.data_ptr(),
                       b1C.data_ptr(), sb1C.data_ptr(),
                       b2C.data_ptr(), sb2C.data_ptr(),
                       d1.data_ptr(), scaleA2.data_ptr(), base.data_ptr(),
                       resBufs.residualPtr, residualInPtr,
                       perRowScale.data_ptr(),
                       wsPtr, wsSize, stream);

    return {base, scaleA2, resBufs.residual};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_rmsnorm_gemm_bf16",
          &gemm_rmsnorm_gemm_bf16,
          "Fused bf16 GEMM + RMSNorm + GEMM via hipblaslt decomposed fused epilogue",
          py::arg("A"),
          py::arg("W1"),
          py::arg("gamma"),
          py::arg("W2"),
          py::arg("eps"),
          py::arg("residual_in"));
    m.def("gemm_rmsnorm_gemm_mxfp8",
          &gemm_rmsnorm_gemm_mxfp8,
          "Full MXFP8 chain: fp8 GEMM1 (MX block-32 input scales) + RMSNorm + fp8 GEMM2",
          py::arg("A_fp8"),
          py::arg("scaleA"),
          py::arg("B1_fp8"),
          py::arg("scaleB1"),
          py::arg("gamma"),
          py::arg("B2"),
          py::arg("scaleB2"),
          py::arg("eps"),
          py::arg("residual_in"));
}

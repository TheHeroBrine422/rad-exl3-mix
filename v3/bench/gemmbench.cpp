// gemmbench.cpp -- hipBLAS GemmEx (f16 or bf16 in/out, f32 accumulate) at the prefill shapes: y[M,N] = x[M,K] W[N,K]^T.
// usage: gemmbench f16|bf16|f16f32|f16bf16 M N K [iters]   (inputs, then output type when it differs)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { auto e_ = (x); if (e_ != 0) { std::printf("fail %s = %d\n", #x, (int)e_); return 1; } } while (0)
int main(int argc, char** argv) {
    const char* m = argv[1];
    const bool bf = !std::strcmp(m, "bf16");
    const hipDataType ct = !std::strcmp(m, "f16f32") ? HIP_R_32F : !std::strcmp(m, "f16bf16") ? HIP_R_16BF : bf ? HIP_R_16BF : HIP_R_16F;
    const long M = atol(argv[2]), N = atol(argv[3]), K = atol(argv[4]);
    const int it = argc > 5 ? atoi(argv[5]) : 20;
    void *x, *w, *y;
    CK(hipMalloc(&x, M * K * 2)); CK(hipMalloc(&w, N * K * 2)); CK(hipMalloc(&y, M * N * 4));
    CK(hipMemset(x, 0x3c, M * K * 2)); CK(hipMemset(w, 0x3c, N * K * 2));
    hipblasHandle_t h; CK(hipblasCreate(&h));
    const hipDataType dt = bf ? HIP_R_16BF : HIP_R_16F;
    const float one = 1.f, zero = 0.f;
    // column-major: Y^T[N,M] = W[N,K] * X^T[K,M]  ->  op(A)=W^T stored KxN col-major (= W row-major), B = X^T
    auto run = [&] {
        return hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &one, w, dt, K, x, dt, K, &zero, y, ct, N,
                             HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    };
    for (int i = 0; i < 3; ++i) CK(run());
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    for (int i = 0; i < it; ++i) run();
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b); ms /= it;
    std::printf("%s M=%ld N=%ld K=%ld: %.3f ms  %.1f TFLOPS\n", argv[1], M, N, K, ms, 2.0 * M * N * K / ms / 1e9);
    return 0;
}


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(250) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ out) {
  float transposed_gemm[8];
  __shared__ float A_shared[4000];
  __shared__ float B_shared[5000];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 2; ++i_outer_inner_init) {
    for (int i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
      transposed_gemm[((i_outer_inner_init * 2) + i_inner_init)] = 0.000000e+00f;
      transposed_gemm[(((i_outer_inner_init * 2) + i_inner_init) + 4)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 10; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(A_shared + ((ax0_ax1_fused_outer_outer * 500) + (((int)threadIdx.x) * 2))) = *(float2*)(A + ((((((((int)blockIdx.x) / 20) * 40000) + (ax0_ax1_fused_outer_outer * 5000)) + ((((int)threadIdx.x) / 50) * 1000)) + (k_outer_outer * 100)) + ((((int)threadIdx.x) % 50) * 2)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 20; ++ax0_ax1_fused_outer_outer_1) {
      B_shared[((ax0_ax1_fused_outer_outer_1 * 250) + ((int)threadIdx.x))] = B[(((((((int)blockIdx.x) % 20) * 50000) + ((((ax0_ax1_fused_outer_outer_1 * 5) + (((int)threadIdx.x) / 50)) >> 1) * 1000)) + (k_outer_outer * 100)) + (((ax0_ax1_fused_outer_outer_1 * 50) + ((int)threadIdx.x)) % 100))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
        for (int k_inner = 0; k_inner < 50; ++k_inner) {
          for (int i_inner = 0; i_inner < 2; ++i_inner) {
            transposed_gemm[((i_outer_inner * 2) + i_inner)] = (transposed_gemm[((i_outer_inner * 2) + i_inner)] + (A_shared[((((((((int)threadIdx.x) / 50) * 400) + (i_outer_inner * 200)) + (i_inner * 100)) + (k_outer_inner * 50)) + k_inner)] * B_shared[((((((int)threadIdx.x) % 50) * 100) + (k_outer_inner * 50)) + k_inner)]));
            transposed_gemm[(((i_outer_inner * 2) + i_inner) + 4)] = (transposed_gemm[(((i_outer_inner * 2) + i_inner) + 4)] + (A_shared[(((((((((int)threadIdx.x) / 50) * 400) + (i_outer_inner * 200)) + (i_inner * 100)) + (k_outer_inner * 50)) + k_inner) + 2000)] * B_shared[((((((int)threadIdx.x) % 50) * 100) + (k_outer_inner * 50)) + k_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner_1 = 0; i_inner_1 < 4; ++i_inner_1) {
    out[((((((((int)blockIdx.x) / 20) * 40000) + ((((int)threadIdx.x) / 50) * 4000)) + (i_inner_1 * 1000)) + ((((int)blockIdx.x) % 20) * 50)) + (((int)threadIdx.x) % 50))] = (transposed_gemm[i_inner_1] + C[((((((((int)blockIdx.x) / 20) * 40000) + ((((int)threadIdx.x) / 50) * 4000)) + (i_inner_1 * 1000)) + ((((int)blockIdx.x) % 20) * 50)) + (((int)threadIdx.x) % 50))]);
    out[(((((((((int)blockIdx.x) / 20) * 40000) + ((((int)threadIdx.x) / 50) * 4000)) + (i_inner_1 * 1000)) + ((((int)blockIdx.x) % 20) * 50)) + (((int)threadIdx.x) % 50)) + 20000)] = (transposed_gemm[(i_inner_1 + 4)] + C[(((((((((int)blockIdx.x) / 20) * 40000) + ((((int)threadIdx.x) / 50) * 4000)) + (i_inner_1 * 1000)) + ((((int)blockIdx.x) % 20) * 50)) + (((int)threadIdx.x) % 50)) + 20000)]);
  }
}



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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ out) {
  float transposed_gemm[16];
  __shared__ float A_shared[256];
  __shared__ float B_shared[512];
  for (int i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
    transposed_gemm[i_inner_init] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 2)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 4)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 6)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 8)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 10)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 12)] = 0.000000e+00f;
    transposed_gemm[(i_inner_init + 14)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 2; ++ax0_ax1_fused_outer_outer) {
      A_shared[((ax0_ax1_fused_outer_outer * 128) + ((int)threadIdx.x))] = A[((((((((int)blockIdx.x) >> 4) * 1024) + (ax0_ax1_fused_outer_outer * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_1) {
      B_shared[((ax0_ax1_fused_outer_outer_1 * 128) + ((int)threadIdx.x))] = B[((((((((int)blockIdx.x) & 15) * 2048) + (ax0_ax1_fused_outer_outer_1 * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 4; ++k_inner) {
        for (int i_inner = 0; i_inner < 2; ++i_inner) {
          transposed_gemm[i_inner] = (transposed_gemm[i_inner] + (A_shared[(((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner)] * B_shared[((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner)]));
          transposed_gemm[(i_inner + 2)] = (transposed_gemm[(i_inner + 2)] + (A_shared[(((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner)] * B_shared[(((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner) + 256)]));
          transposed_gemm[(i_inner + 4)] = (transposed_gemm[(i_inner + 4)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 64)] * B_shared[((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner)]));
          transposed_gemm[(i_inner + 6)] = (transposed_gemm[(i_inner + 6)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 64)] * B_shared[(((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner) + 256)]));
          transposed_gemm[(i_inner + 8)] = (transposed_gemm[(i_inner + 8)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 128)] * B_shared[((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner)]));
          transposed_gemm[(i_inner + 10)] = (transposed_gemm[(i_inner + 10)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 128)] * B_shared[(((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner) + 256)]));
          transposed_gemm[(i_inner + 12)] = (transposed_gemm[(i_inner + 12)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 192)] * B_shared[((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner)]));
          transposed_gemm[(i_inner + 14)] = (transposed_gemm[(i_inner + 14)] + (A_shared[((((((((int)threadIdx.x) >> 5) * 16) + (i_inner * 8)) + (k_outer_inner * 4)) + k_inner) + 192)] * B_shared[(((((((int)threadIdx.x) & 31) * 8) + (k_outer_inner * 4)) + k_inner) + 256)]));
        }
      }
    }
  }
  for (int i_inner_1 = 0; i_inner_1 < 2; ++i_inner_1) {
    out[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31))] = (transposed_gemm[i_inner_1] + C[((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31))]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 32)] = (transposed_gemm[(i_inner_1 + 2)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 32)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 8192)] = (transposed_gemm[(i_inner_1 + 4)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 8192)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 8224)] = (transposed_gemm[(i_inner_1 + 6)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 8224)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 16384)] = (transposed_gemm[(i_inner_1 + 8)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 16384)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 16416)] = (transposed_gemm[(i_inner_1 + 10)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 16416)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 24576)] = (transposed_gemm[(i_inner_1 + 12)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 24576)]);
    out[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 24608)] = (transposed_gemm[(i_inner_1 + 14)] + C[(((((((((int)blockIdx.x) >> 4) * 32768) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 15) * 64)) + (((int)threadIdx.x) & 31)) + 24608)]);
  }
}


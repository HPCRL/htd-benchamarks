
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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ out) {
  float transposed_gemm[8];
  __shared__ float A_shared[256];
  __shared__ float B_shared[256];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 4; ++i_outer_inner_init) {
    for (int i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
      transposed_gemm[((i_outer_inner_init * 2) + i_inner_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      A_shared[((ax0_ax1_fused_outer_outer * 32) + ((int)threadIdx.x))] = A[((((((((int)blockIdx.x) >> 4) * 4096) + (ax0_ax1_fused_outer_outer * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_1) {
      *(float2*)(B_shared + ((ax0_ax1_fused_outer_outer_1 * 64) + (((int)threadIdx.x) * 2))) = *(float2*)(B + ((((((((int)blockIdx.x) & 15) * 4096) + (ax0_ax1_fused_outer_outer_1 * 1024)) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    }
    __syncthreads();
    for (int i_outer_inner = 0; i_outer_inner < 4; ++i_outer_inner) {
      for (int k_inner = 0; k_inner < 16; ++k_inner) {
        for (int i_inner = 0; i_inner < 2; ++i_inner) {
          transposed_gemm[((i_outer_inner * 2) + i_inner)] = (transposed_gemm[((i_outer_inner * 2) + i_inner)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 32)) + (i_inner * 16)) + k_inner)] * B_shared[(((((int)threadIdx.x) & 15) * 16) + k_inner)]));
        }
      }
    }
  }
  for (int i_inner_1 = 0; i_inner_1 < 8; ++i_inner_1) {
    out[((((((((int)blockIdx.x) >> 4) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner_1 * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner_1] + C[((((((((int)blockIdx.x) >> 4) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner_1 * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))]);
  }
}


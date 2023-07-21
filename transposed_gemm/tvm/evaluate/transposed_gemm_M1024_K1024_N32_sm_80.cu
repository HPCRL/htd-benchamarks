
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
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  for (int j_outer_inner_init = 0; j_outer_inner_init < 4; ++j_outer_inner_init) {
    transposed_gemm[j_outer_inner_init] = 0.000000e+00f;
    transposed_gemm[(j_outer_inner_init + 4)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 16; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(A_shared + ((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((((int)blockIdx.x) >> 1) * 16384) + (ax0_ax1_fused_outer_outer * 1024)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      *(float4*)(B_shared + ((ax0_ax1_fused_outer_outer_1 * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(B + (((((((int)blockIdx.x) & 1) * 16384) + (ax0_ax1_fused_outer_outer_1 * 1024)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int j_outer_inner = 0; j_outer_inner < 4; ++j_outer_inner) {
        for (int k_inner = 0; k_inner < 8; ++k_inner) {
          transposed_gemm[j_outer_inner] = (transposed_gemm[j_outer_inner] + (A_shared[((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner)] * B_shared[(((((((int)threadIdx.x) & 1) * 512) + (j_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner)]));
          transposed_gemm[(j_outer_inner + 4)] = (transposed_gemm[(j_outer_inner + 4)] + (A_shared[((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner)] * B_shared[((((((((int)threadIdx.x) & 1) * 512) + (j_outer_inner * 128)) + (k_outer_inner * 8)) + k_inner) + 1024)]));
        }
      }
    }
  }
  for (int j_inner = 0; j_inner < 4; ++j_inner) {
    out[((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner)] = (transposed_gemm[j_inner] + C[((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner)]);
    out[(((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner) + 8)] = (transposed_gemm[(j_inner + 4)] + C[(((((((((int)blockIdx.x) >> 1) * 512) + ((((int)threadIdx.x) >> 1) * 32)) + ((((int)blockIdx.x) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner) + 8)]);
  }
}


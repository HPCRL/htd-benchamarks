
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ out) {
  float transposed_gemm[32];
  __shared__ float A_shared[128];
  __shared__ float B_shared[1024];
  for (int i_inner_init = 0; i_inner_init < 16; ++i_inner_init) {
    transposed_gemm[(i_inner_init * 2)] = 0.000000e+00f;
    transposed_gemm[((i_inner_init * 2) + 1)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((((int)blockIdx.x) >> 3) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    A_shared[(((int)threadIdx.x) + 64)] = A[((((((((int)blockIdx.x) >> 3) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1024)];
    *(float4*)(B_shared + (((int)threadIdx.x) * 4)) = *(float4*)(B + (((((((int)blockIdx.x) & 7) * 16384) + ((((int)threadIdx.x) >> 1) * 128)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(B + ((((((((int)blockIdx.x) & 7) * 16384) + ((((int)threadIdx.x) >> 1) * 128)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 4096));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(B + ((((((((int)blockIdx.x) & 7) * 16384) + ((((int)threadIdx.x) >> 1) * 128)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(B + ((((((((int)blockIdx.x) & 7) * 16384) + ((((int)threadIdx.x) >> 1) * 128)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 12288));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 4; ++k_inner) {
        for (int i_inner = 0; i_inner < 16; ++i_inner) {
          transposed_gemm[(i_inner * 2)] = (transposed_gemm[(i_inner * 2)] + (A_shared[(((i_inner * 8) + (k_outer_inner * 4)) + k_inner)] * B_shared[(((((int)threadIdx.x) * 16) + (k_outer_inner * 4)) + k_inner)]));
          transposed_gemm[((i_inner * 2) + 1)] = (transposed_gemm[((i_inner * 2) + 1)] + (A_shared[(((i_inner * 8) + (k_outer_inner * 4)) + k_inner)] * B_shared[((((((int)threadIdx.x) * 16) + (k_outer_inner * 4)) + k_inner) + 8)]));
        }
      }
    }
  }
  for (int i_inner_1 = 0; i_inner_1 < 16; ++i_inner_1) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      out[((((((((int)blockIdx.x) >> 3) * 16384) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + (((int)threadIdx.x) * 2)) + j_inner)] = (transposed_gemm[((i_inner_1 * 2) + j_inner)] + C[((((((((int)blockIdx.x) >> 3) * 16384) + (i_inner_1 * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + (((int)threadIdx.x) * 2)) + j_inner)]);
    }
  }
}



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
  float transposed_gemm[2];
  __shared__ float A_shared[32];
  __shared__ float B_shared[128];
  transposed_gemm[0] = 0.000000e+00f;
  transposed_gemm[1] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((((int)blockIdx.x) >> 6) * 4096) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_outer_outer * 8)) + (((int)threadIdx.x) & 7))];
    *(float2*)(B_shared + (((int)threadIdx.x) * 2)) = *(float2*)(B + (((((((int)blockIdx.x) & 63) * 16384) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(float2*)(B_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(B + ((((((((int)blockIdx.x) & 63) * 16384) + ((((int)threadIdx.x) >> 2) * 1024)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
    __syncthreads();
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 4) * 16)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 8)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
    transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
    transposed_gemm[1] = (transposed_gemm[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    out[((((((((int)blockIdx.x) >> 6) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 63) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner] + C[((((((((int)blockIdx.x) >> 6) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 63) * 16)) + (((int)threadIdx.x) & 15))]);
  }
}


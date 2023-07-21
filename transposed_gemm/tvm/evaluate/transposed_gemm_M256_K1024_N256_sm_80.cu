
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
  transposed_gemm[0] = 0.000000e+00f;
  transposed_gemm[1] = 0.000000e+00f;
  transposed_gemm[2] = 0.000000e+00f;
  transposed_gemm[3] = 0.000000e+00f;
  transposed_gemm[4] = 0.000000e+00f;
  transposed_gemm[5] = 0.000000e+00f;
  transposed_gemm[6] = 0.000000e+00f;
  transposed_gemm[7] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + ((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 8192));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9216));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 10240));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 11264));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 12288));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1664)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13312));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 14336));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1920)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 15360));
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      *(float2*)(B_shared + ((ax0_ax1_fused_outer_outer * 64) + (((int)threadIdx.x) * 2))) = *(float2*)(B + ((((((((int)blockIdx.x) & 15) * 16384) + ((ax0_ax1_fused_outer_outer >> 1) * 1024)) + (k_outer_outer * 128)) + ((ax0_ax1_fused_outer_outer & 1) * 64)) + (((int)threadIdx.x) * 2)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int i_outer_inner = 0; i_outer_inner < 8; ++i_outer_inner) {
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8))] * B_shared[(((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8))]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 1)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 1)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 2)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 2)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 3)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 3)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 4)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 4)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 5)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 5)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 6)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 6)]));
        transposed_gemm[i_outer_inner] = (transposed_gemm[i_outer_inner] + (A_shared[(((((((int)threadIdx.x) >> 4) * 1024) + (i_outer_inner * 128)) + (k_outer_inner * 8)) + 7)] * B_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 8)) + 7)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    out[((((((((int)blockIdx.x) >> 4) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner] + C[((((((((int)blockIdx.x) >> 4) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15))]);
  }
}


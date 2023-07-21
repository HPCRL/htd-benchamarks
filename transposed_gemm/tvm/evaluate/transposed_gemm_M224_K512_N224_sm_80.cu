
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
  float transposed_gemm[4];
  __shared__ float A_shared[8192];
  __shared__ float B_shared[4096];
  transposed_gemm[0] = 0.000000e+00f;
  transposed_gemm[1] = 0.000000e+00f;
  transposed_gemm[2] = 0.000000e+00f;
  transposed_gemm[3] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x))];
    A_shared[(((int)threadIdx.x) + 128)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 128)];
    A_shared[(((int)threadIdx.x) + 256)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 512)];
    A_shared[(((int)threadIdx.x) + 384)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 640)];
    A_shared[(((int)threadIdx.x) + 512)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1024)];
    A_shared[(((int)threadIdx.x) + 640)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1152)];
    A_shared[(((int)threadIdx.x) + 768)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1536)];
    A_shared[(((int)threadIdx.x) + 896)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1664)];
    A_shared[(((int)threadIdx.x) + 1024)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2048)];
    A_shared[(((int)threadIdx.x) + 1152)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2176)];
    A_shared[(((int)threadIdx.x) + 1280)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2560)];
    A_shared[(((int)threadIdx.x) + 1408)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2688)];
    A_shared[(((int)threadIdx.x) + 1536)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3072)];
    A_shared[(((int)threadIdx.x) + 1664)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3200)];
    A_shared[(((int)threadIdx.x) + 1792)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3584)];
    A_shared[(((int)threadIdx.x) + 1920)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3712)];
    A_shared[(((int)threadIdx.x) + 2048)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4096)];
    A_shared[(((int)threadIdx.x) + 2176)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4224)];
    A_shared[(((int)threadIdx.x) + 2304)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4608)];
    A_shared[(((int)threadIdx.x) + 2432)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4736)];
    A_shared[(((int)threadIdx.x) + 2560)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5120)];
    A_shared[(((int)threadIdx.x) + 2688)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5248)];
    A_shared[(((int)threadIdx.x) + 2816)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5632)];
    A_shared[(((int)threadIdx.x) + 2944)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5760)];
    A_shared[(((int)threadIdx.x) + 3072)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6144)];
    A_shared[(((int)threadIdx.x) + 3200)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6272)];
    A_shared[(((int)threadIdx.x) + 3328)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6656)];
    A_shared[(((int)threadIdx.x) + 3456)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6784)];
    A_shared[(((int)threadIdx.x) + 3584)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7168)];
    A_shared[(((int)threadIdx.x) + 3712)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7296)];
    A_shared[(((int)threadIdx.x) + 3840)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7680)];
    A_shared[(((int)threadIdx.x) + 3968)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7808)];
    A_shared[(((int)threadIdx.x) + 4096)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8192)];
    A_shared[(((int)threadIdx.x) + 4224)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8320)];
    A_shared[(((int)threadIdx.x) + 4352)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8704)];
    A_shared[(((int)threadIdx.x) + 4480)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8832)];
    A_shared[(((int)threadIdx.x) + 4608)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9216)];
    A_shared[(((int)threadIdx.x) + 4736)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9344)];
    A_shared[(((int)threadIdx.x) + 4864)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9728)];
    A_shared[(((int)threadIdx.x) + 4992)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9856)];
    A_shared[(((int)threadIdx.x) + 5120)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10240)];
    A_shared[(((int)threadIdx.x) + 5248)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10368)];
    A_shared[(((int)threadIdx.x) + 5376)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10752)];
    A_shared[(((int)threadIdx.x) + 5504)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10880)];
    A_shared[(((int)threadIdx.x) + 5632)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11264)];
    A_shared[(((int)threadIdx.x) + 5760)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11392)];
    A_shared[(((int)threadIdx.x) + 5888)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11776)];
    A_shared[(((int)threadIdx.x) + 6016)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11904)];
    A_shared[(((int)threadIdx.x) + 6144)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12288)];
    A_shared[(((int)threadIdx.x) + 6272)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12416)];
    A_shared[(((int)threadIdx.x) + 6400)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12800)];
    A_shared[(((int)threadIdx.x) + 6528)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12928)];
    A_shared[(((int)threadIdx.x) + 6656)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13312)];
    A_shared[(((int)threadIdx.x) + 6784)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13440)];
    A_shared[(((int)threadIdx.x) + 6912)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13824)];
    A_shared[(((int)threadIdx.x) + 7040)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13952)];
    A_shared[(((int)threadIdx.x) + 7168)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14336)];
    A_shared[(((int)threadIdx.x) + 7296)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14464)];
    A_shared[(((int)threadIdx.x) + 7424)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14848)];
    A_shared[(((int)threadIdx.x) + 7552)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14976)];
    A_shared[(((int)threadIdx.x) + 7680)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15360)];
    A_shared[(((int)threadIdx.x) + 7808)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15488)];
    A_shared[(((int)threadIdx.x) + 7936)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15872)];
    A_shared[(((int)threadIdx.x) + 8064)] = A[(((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 16000)];
    *(float4*)(B_shared + (((int)threadIdx.x) * 4)) = *(float4*)(B + (((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 1024));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 2048));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 3072));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 4096));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 5120));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 6144));
    *(float4*)(B_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(B + ((((((((int)blockIdx.x) % 14) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) & 63) * 4)) + 7168));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 256; ++k_outer_inner) {
      transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 1024) + k_outer_inner)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + k_outer_inner)]));
      transposed_gemm[1] = (transposed_gemm[1] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + k_outer_inner) + 256)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + k_outer_inner)]));
      transposed_gemm[2] = (transposed_gemm[2] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + k_outer_inner) + 512)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + k_outer_inner)]));
      transposed_gemm[3] = (transposed_gemm[3] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + k_outer_inner) + 768)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + k_outer_inner)]));
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    out[((((((((int)blockIdx.x) / 14) * 7168) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner] + C[((((((((int)blockIdx.x) / 14) * 7168) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15))]);
  }
}


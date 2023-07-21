
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
  float transposed_gemm[16];
  __shared__ float A_shared[256];
  __shared__ float B_shared[128];
  transposed_gemm[0] = 0.000000e+00f;
  transposed_gemm[1] = 0.000000e+00f;
  transposed_gemm[2] = 0.000000e+00f;
  transposed_gemm[3] = 0.000000e+00f;
  transposed_gemm[4] = 0.000000e+00f;
  transposed_gemm[5] = 0.000000e+00f;
  transposed_gemm[6] = 0.000000e+00f;
  transposed_gemm[7] = 0.000000e+00f;
  transposed_gemm[8] = 0.000000e+00f;
  transposed_gemm[9] = 0.000000e+00f;
  transposed_gemm[10] = 0.000000e+00f;
  transposed_gemm[11] = 0.000000e+00f;
  transposed_gemm[12] = 0.000000e+00f;
  transposed_gemm[13] = 0.000000e+00f;
  transposed_gemm[14] = 0.000000e+00f;
  transposed_gemm[15] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) >> 6) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 6) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192));
    *(float2*)(B_shared + (((int)threadIdx.x) * 2)) = *(float2*)(B + (((((((int)blockIdx.x) & 63) * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    *(float2*)(B_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(B + ((((((((int)blockIdx.x) & 63) * 8192) + ((((int)threadIdx.x) >> 2) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096));
    __syncthreads();
    for (int i_outer_inner = 0; i_outer_inner < 2; ++i_outer_inner) {
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[(((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64))] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 8)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 16)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 24)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 32)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 40)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 48)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 56)] * B_shared[((((int)threadIdx.x) & 15) * 8)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 1)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 9)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 17)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 25)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 33)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 41)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 49)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 57)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 1)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 2)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 10)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 18)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 26)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 34)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 42)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 50)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 58)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 2)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 3)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 11)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 19)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 27)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 35)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 43)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 51)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 59)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 3)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 4)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 12)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 20)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 28)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 36)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 44)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 52)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 60)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 4)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 5)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 13)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 21)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 29)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 37)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 45)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 53)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 61)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 5)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 6)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 14)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 22)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 30)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 38)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 46)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 54)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 62)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 6)]));
      transposed_gemm[(i_outer_inner * 8)] = (transposed_gemm[(i_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 7)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 1)] = (transposed_gemm[((i_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 15)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 2)] = (transposed_gemm[((i_outer_inner * 8) + 2)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 23)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 3)] = (transposed_gemm[((i_outer_inner * 8) + 3)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 31)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 4)] = (transposed_gemm[((i_outer_inner * 8) + 4)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 39)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 5)] = (transposed_gemm[((i_outer_inner * 8) + 5)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 47)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 6)] = (transposed_gemm[((i_outer_inner * 8) + 6)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 55)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
      transposed_gemm[((i_outer_inner * 8) + 7)] = (transposed_gemm[((i_outer_inner * 8) + 7)] + (A_shared[((((((int)threadIdx.x) >> 4) * 128) + (i_outer_inner * 64)) + 63)] * B_shared[(((((int)threadIdx.x) & 15) * 8) + 7)]));
    }
  }
  for (int i_inner = 0; i_inner < 16; ++i_inner) {
    out[((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 63) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner] + C[((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 4) * 16384)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 63) * 16)) + (((int)threadIdx.x) & 15))]);
  }
}



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
extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, float* __restrict__ out) {
  float transposed_gemm[1];
  __shared__ float A_shared[192];
  __shared__ float B_shared[64];
  transposed_gemm[0] = 0.000000e+00f;
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + ((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)));
  B_shared[((int)threadIdx.x)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15))];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1536)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 16)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1552)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 32)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1568)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 48));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 48)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1584)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 64));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 64)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1600)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 80));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 80)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1616)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 96));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 96)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1632)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 112));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 112)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1648)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 128));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 128)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1664)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 144));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 144)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1680)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 160));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 160)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1696)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 176));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 176)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1712)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 192));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 192)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1728)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 208));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 208)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1744)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 224));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 224)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1760)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 240));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 240)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1776)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 256));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 256)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1792)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 272));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 272)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1808)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 288));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 288)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1824)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 304));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 304)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1840)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 320));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 320)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1856)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 336));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 336)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1872)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 352));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 352)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1888)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 368));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 368)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1904)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 384));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 384)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1920)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 400));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 400)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1936)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 416));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 416)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1952)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 432));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 432)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1968)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 448));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 448)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 1984)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 464));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 464)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 2000)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 480));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 480)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 2016)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  __syncthreads();
  *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) / 30) * 6144) + ((((int)threadIdx.x) >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 4)) + 496));
  B_shared[((int)threadIdx.x)] = B[(((((((int)blockIdx.x) % 30) * 2048) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)threadIdx.x) & 15)) + 496)];
  if (((int)threadIdx.x) < 16) {
    B_shared[(((int)threadIdx.x) + 48)] = B[((((((int)blockIdx.x) % 30) * 2048) + ((int)threadIdx.x)) + 2032)];
  }
  __syncthreads();
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((int)threadIdx.x) >> 2) * 16)] * B_shared[((((int)threadIdx.x) & 3) * 16)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 1)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 1)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 2)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 2)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 3)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 3)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 4)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 4)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 5)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 5)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 6)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 6)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 7)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 7)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 8)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 8)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 9)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 9)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 10)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 10)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 11)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 11)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 12)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 12)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 13)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 13)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 14)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 14)]));
  transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 2) * 16) + 15)] * B_shared[(((((int)threadIdx.x) & 3) * 16) + 15)]));
  out[(((((((int)blockIdx.x) / 30) * 1440) + ((((int)threadIdx.x) >> 2) * 120)) + ((((int)blockIdx.x) % 30) * 4)) + (((int)threadIdx.x) & 3))] = (transposed_gemm[0] + C[(((((((int)blockIdx.x) / 30) * 1440) + ((((int)threadIdx.x) >> 2) * 120)) + ((((int)blockIdx.x) % 30) * 4)) + (((int)threadIdx.x) & 3))]);
}


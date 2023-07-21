
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
  __shared__ float A_shared[4096];
  __shared__ float B_shared[4096];
  transposed_gemm[0] = 0.000000e+00f;
  transposed_gemm[4] = 0.000000e+00f;
  transposed_gemm[1] = 0.000000e+00f;
  transposed_gemm[5] = 0.000000e+00f;
  transposed_gemm[2] = 0.000000e+00f;
  transposed_gemm[6] = 0.000000e+00f;
  transposed_gemm[3] = 0.000000e+00f;
  transposed_gemm[7] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + ((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 128));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 1024));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 1152));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 2048));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 2176));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 3072));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 3200));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 4096));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1152)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 4224));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 5120));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1408)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 5248));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 6144));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1664)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 6272));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 7168));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1920)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 7296));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 8192));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2176)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 8320));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 9216));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2432)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 9344));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 10240));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2688)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 10368));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 11264));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 2944)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 11392));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 12288));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3200)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 12416));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3328)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 13312));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3456)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 13440));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 14336));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3712)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 14464));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3840)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 15360));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 3968)) = *(float4*)(A + (((((((int)blockIdx.x) / 14) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 15488));
    B_shared[((int)threadIdx.x)] = B[((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x))];
    B_shared[(((int)threadIdx.x) + 32)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 32)];
    B_shared[(((int)threadIdx.x) + 64)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 64)];
    B_shared[(((int)threadIdx.x) + 96)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 96)];
    B_shared[(((int)threadIdx.x) + 128)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 128)];
    B_shared[(((int)threadIdx.x) + 160)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 160)];
    B_shared[(((int)threadIdx.x) + 192)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 192)];
    B_shared[(((int)threadIdx.x) + 224)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 224)];
    B_shared[(((int)threadIdx.x) + 256)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1024)];
    B_shared[(((int)threadIdx.x) + 288)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1056)];
    B_shared[(((int)threadIdx.x) + 320)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1088)];
    B_shared[(((int)threadIdx.x) + 352)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1120)];
    B_shared[(((int)threadIdx.x) + 384)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1152)];
    B_shared[(((int)threadIdx.x) + 416)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1184)];
    B_shared[(((int)threadIdx.x) + 448)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1216)];
    B_shared[(((int)threadIdx.x) + 480)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 1248)];
    B_shared[(((int)threadIdx.x) + 512)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2048)];
    B_shared[(((int)threadIdx.x) + 544)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2080)];
    B_shared[(((int)threadIdx.x) + 576)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2112)];
    B_shared[(((int)threadIdx.x) + 608)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2144)];
    B_shared[(((int)threadIdx.x) + 640)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2176)];
    B_shared[(((int)threadIdx.x) + 672)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2208)];
    B_shared[(((int)threadIdx.x) + 704)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2240)];
    B_shared[(((int)threadIdx.x) + 736)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 2272)];
    B_shared[(((int)threadIdx.x) + 768)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3072)];
    B_shared[(((int)threadIdx.x) + 800)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3104)];
    B_shared[(((int)threadIdx.x) + 832)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3136)];
    B_shared[(((int)threadIdx.x) + 864)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3168)];
    B_shared[(((int)threadIdx.x) + 896)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3200)];
    B_shared[(((int)threadIdx.x) + 928)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3232)];
    B_shared[(((int)threadIdx.x) + 960)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3264)];
    B_shared[(((int)threadIdx.x) + 992)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 3296)];
    B_shared[(((int)threadIdx.x) + 1024)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4096)];
    B_shared[(((int)threadIdx.x) + 1056)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4128)];
    B_shared[(((int)threadIdx.x) + 1088)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4160)];
    B_shared[(((int)threadIdx.x) + 1120)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4192)];
    B_shared[(((int)threadIdx.x) + 1152)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4224)];
    B_shared[(((int)threadIdx.x) + 1184)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4256)];
    B_shared[(((int)threadIdx.x) + 1216)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4288)];
    B_shared[(((int)threadIdx.x) + 1248)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 4320)];
    B_shared[(((int)threadIdx.x) + 1280)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5120)];
    B_shared[(((int)threadIdx.x) + 1312)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5152)];
    B_shared[(((int)threadIdx.x) + 1344)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5184)];
    B_shared[(((int)threadIdx.x) + 1376)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5216)];
    B_shared[(((int)threadIdx.x) + 1408)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5248)];
    B_shared[(((int)threadIdx.x) + 1440)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5280)];
    B_shared[(((int)threadIdx.x) + 1472)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5312)];
    B_shared[(((int)threadIdx.x) + 1504)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 5344)];
    B_shared[(((int)threadIdx.x) + 1536)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6144)];
    B_shared[(((int)threadIdx.x) + 1568)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6176)];
    B_shared[(((int)threadIdx.x) + 1600)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6208)];
    B_shared[(((int)threadIdx.x) + 1632)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6240)];
    B_shared[(((int)threadIdx.x) + 1664)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6272)];
    B_shared[(((int)threadIdx.x) + 1696)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6304)];
    B_shared[(((int)threadIdx.x) + 1728)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6336)];
    B_shared[(((int)threadIdx.x) + 1760)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 6368)];
    B_shared[(((int)threadIdx.x) + 1792)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7168)];
    B_shared[(((int)threadIdx.x) + 1824)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7200)];
    B_shared[(((int)threadIdx.x) + 1856)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7232)];
    B_shared[(((int)threadIdx.x) + 1888)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7264)];
    B_shared[(((int)threadIdx.x) + 1920)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7296)];
    B_shared[(((int)threadIdx.x) + 1952)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7328)];
    B_shared[(((int)threadIdx.x) + 1984)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7360)];
    B_shared[(((int)threadIdx.x) + 2016)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 7392)];
    B_shared[(((int)threadIdx.x) + 2048)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8192)];
    B_shared[(((int)threadIdx.x) + 2080)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8224)];
    B_shared[(((int)threadIdx.x) + 2112)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8256)];
    B_shared[(((int)threadIdx.x) + 2144)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8288)];
    B_shared[(((int)threadIdx.x) + 2176)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8320)];
    B_shared[(((int)threadIdx.x) + 2208)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8352)];
    B_shared[(((int)threadIdx.x) + 2240)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8384)];
    B_shared[(((int)threadIdx.x) + 2272)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 8416)];
    B_shared[(((int)threadIdx.x) + 2304)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9216)];
    B_shared[(((int)threadIdx.x) + 2336)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9248)];
    B_shared[(((int)threadIdx.x) + 2368)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9280)];
    B_shared[(((int)threadIdx.x) + 2400)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9312)];
    B_shared[(((int)threadIdx.x) + 2432)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9344)];
    B_shared[(((int)threadIdx.x) + 2464)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9376)];
    B_shared[(((int)threadIdx.x) + 2496)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9408)];
    B_shared[(((int)threadIdx.x) + 2528)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 9440)];
    B_shared[(((int)threadIdx.x) + 2560)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10240)];
    B_shared[(((int)threadIdx.x) + 2592)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10272)];
    B_shared[(((int)threadIdx.x) + 2624)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10304)];
    B_shared[(((int)threadIdx.x) + 2656)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10336)];
    B_shared[(((int)threadIdx.x) + 2688)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10368)];
    B_shared[(((int)threadIdx.x) + 2720)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10400)];
    B_shared[(((int)threadIdx.x) + 2752)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10432)];
    B_shared[(((int)threadIdx.x) + 2784)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 10464)];
    B_shared[(((int)threadIdx.x) + 2816)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11264)];
    B_shared[(((int)threadIdx.x) + 2848)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11296)];
    B_shared[(((int)threadIdx.x) + 2880)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11328)];
    B_shared[(((int)threadIdx.x) + 2912)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11360)];
    B_shared[(((int)threadIdx.x) + 2944)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11392)];
    B_shared[(((int)threadIdx.x) + 2976)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11424)];
    B_shared[(((int)threadIdx.x) + 3008)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11456)];
    B_shared[(((int)threadIdx.x) + 3040)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 11488)];
    B_shared[(((int)threadIdx.x) + 3072)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12288)];
    B_shared[(((int)threadIdx.x) + 3104)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12320)];
    B_shared[(((int)threadIdx.x) + 3136)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12352)];
    B_shared[(((int)threadIdx.x) + 3168)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12384)];
    B_shared[(((int)threadIdx.x) + 3200)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12416)];
    B_shared[(((int)threadIdx.x) + 3232)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12448)];
    B_shared[(((int)threadIdx.x) + 3264)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12480)];
    B_shared[(((int)threadIdx.x) + 3296)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 12512)];
    B_shared[(((int)threadIdx.x) + 3328)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13312)];
    B_shared[(((int)threadIdx.x) + 3360)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13344)];
    B_shared[(((int)threadIdx.x) + 3392)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13376)];
    B_shared[(((int)threadIdx.x) + 3424)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13408)];
    B_shared[(((int)threadIdx.x) + 3456)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13440)];
    B_shared[(((int)threadIdx.x) + 3488)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13472)];
    B_shared[(((int)threadIdx.x) + 3520)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13504)];
    B_shared[(((int)threadIdx.x) + 3552)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 13536)];
    B_shared[(((int)threadIdx.x) + 3584)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14336)];
    B_shared[(((int)threadIdx.x) + 3616)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14368)];
    B_shared[(((int)threadIdx.x) + 3648)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14400)];
    B_shared[(((int)threadIdx.x) + 3680)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14432)];
    B_shared[(((int)threadIdx.x) + 3712)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14464)];
    B_shared[(((int)threadIdx.x) + 3744)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14496)];
    B_shared[(((int)threadIdx.x) + 3776)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14528)];
    B_shared[(((int)threadIdx.x) + 3808)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 14560)];
    B_shared[(((int)threadIdx.x) + 3840)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15360)];
    B_shared[(((int)threadIdx.x) + 3872)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15392)];
    B_shared[(((int)threadIdx.x) + 3904)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15424)];
    B_shared[(((int)threadIdx.x) + 3936)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15456)];
    B_shared[(((int)threadIdx.x) + 3968)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15488)];
    B_shared[(((int)threadIdx.x) + 4000)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15520)];
    B_shared[(((int)threadIdx.x) + 4032)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15552)];
    B_shared[(((int)threadIdx.x) + 4064)] = B[(((((((int)blockIdx.x) % 14) * 16384) + (k_outer_outer * 256)) + ((int)threadIdx.x)) + 15584)];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 64; ++k_outer_inner) {
      transposed_gemm[0] = (transposed_gemm[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4))] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[4] = (transposed_gemm[4] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2048)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 1)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[4] = (transposed_gemm[4] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2049)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[4] = (transposed_gemm[4] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2050)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[0] = (transposed_gemm[0] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 3)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[4] = (transposed_gemm[4] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2051)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[1] = (transposed_gemm[1] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 256)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[5] = (transposed_gemm[5] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2304)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[1] = (transposed_gemm[1] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 257)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[5] = (transposed_gemm[5] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2305)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[1] = (transposed_gemm[1] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 258)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[5] = (transposed_gemm[5] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2306)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[1] = (transposed_gemm[1] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 259)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[5] = (transposed_gemm[5] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2307)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[2] = (transposed_gemm[2] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 512)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[6] = (transposed_gemm[6] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2560)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[2] = (transposed_gemm[2] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 513)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[6] = (transposed_gemm[6] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2561)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[2] = (transposed_gemm[2] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 514)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[6] = (transposed_gemm[6] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2562)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[2] = (transposed_gemm[2] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 515)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[6] = (transposed_gemm[6] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2563)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[3] = (transposed_gemm[3] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 768)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[7] = (transposed_gemm[7] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2816)] * B_shared[(((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4))]));
      transposed_gemm[3] = (transposed_gemm[3] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 769)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[7] = (transposed_gemm[7] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2817)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 1)]));
      transposed_gemm[3] = (transposed_gemm[3] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 770)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[7] = (transposed_gemm[7] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2818)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 2)]));
      transposed_gemm[3] = (transposed_gemm[3] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 771)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
      transposed_gemm[7] = (transposed_gemm[7] + (A_shared[((((((int)threadIdx.x) >> 4) * 1024) + (k_outer_inner * 4)) + 2819)] * B_shared[((((((int)threadIdx.x) & 15) * 256) + (k_outer_inner * 4)) + 3)]));
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    out[((((((((int)blockIdx.x) / 14) * 3584) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15))] = (transposed_gemm[i_inner] + C[((((((((int)blockIdx.x) / 14) * 3584) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15))]);
    out[(((((((((int)blockIdx.x) / 14) * 3584) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15)) + 1792)] = (transposed_gemm[(i_inner + 4)] + C[(((((((((int)blockIdx.x) / 14) * 3584) + ((((int)threadIdx.x) >> 4) * 896)) + (i_inner * 224)) + ((((int)blockIdx.x) % 14) * 16)) + (((int)threadIdx.x) & 15)) + 1792)]);
  }
}


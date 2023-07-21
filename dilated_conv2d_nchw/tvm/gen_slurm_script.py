arr = [
    [1, 28, 28, 512, 512, 3, 3, 1, 1, 2],
    [1, 28, 28, 512, 512, 3, 3, 1, 1, 3],
    [1, 14, 14, 256, 256, 3, 3, 1, 1, 2],
    [1, 14, 14, 256, 256, 3, 3, 1, 1, 3],
    [1, 224, 224, 64, 3, 3, 3, 1, 1, 2],
    [1, 224, 224, 64, 3, 3, 3, 1, 1, 3],
    [1, 224, 224, 64, 3, 7, 7, 2, 3, 2],
    [1, 224, 224, 64, 3, 7, 7, 2, 3, 3],
    [1, 68, 68, 256, 128, 3, 3, 1, 1, 2],
    [1, 68, 68, 256, 128, 3, 3, 1, 1, 3],
    [1, 68, 68, 512, 256, 3, 3, 1, 1, 2],
    [1, 68, 68, 512, 256, 3, 3, 1, 1, 3],
    [1, 544, 544, 32, 3, 3, 3, 1, 1, 2],
    [1, 544, 544, 32, 3, 3, 3, 1, 1, 3],
]

mode = "timing"
tune_number = 1000 if mode == "tune" else ""
if mode != "timing":
    with open(f"all_dilated_conv2d_nchw_{mode}.sh", 'w') as f:
        for item in arr:
            f.write(f"sbatch {mode}.slurm {' '.join(map(str, item))} {tune_number}")
else: 
    with open(f"all_dilated_conv2d_nchw_{mode}.sh", 'w') as f:
        for item in arr:
            N, H, W, CO, CI, KH, KW, stride, pad, dilation = item
            f.write(f"echo \"Running dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80\"\n")
            f.write(f"cd /uufs/chpc.utah.edu/common/home/u1419116/projects/hlt/codegen/tvm/src/dilated_conv2d_nchw && bash {mode}.slurm {' '.join(map(str, item))} &> ../../../results/dilated_conv2d_nchw/tvm/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.txt\n")
            f.write(f"cd /uufs/chpc.utah.edu/common/home/u1419116/projects/hlt/codegen && cp auto_tuner_cpp/dilated_conv2d_nchw/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{stride}_padding_{pad}_dilation_{dilation}_sm_80.cpp main.cpp && cd build && rm -rf * && cmake .. && make && ./codegen && cd ../test/dilated_conv2d_nchw/bash_scripts && bash dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.sh &> ../../../results/dilated_conv2d_nchw/autotuner/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.txt && cd ../../../tvm/src/dilated_conv2d_nchw/\n")
            f.write(f"cd /uufs/chpc.utah.edu/common/home/u1419116/projects/hlt/codegen && cp tvm/ansor_schedule/dilated_conv2d_nchw/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_strides_{stride}_padding_{pad}_dilation_{dilation}_sm_80.cpp main.cpp && cd build && rm -rf * && cmake .. && make && ./codegen && cd ../test/dilated_conv2d_nchw/bash_scripts && bash dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.sh &> ../../../results/dilated_conv2d_nchw/same_schedule/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.txt && cd ../../../tvm/src/dilated_conv2d_nchw/\n")

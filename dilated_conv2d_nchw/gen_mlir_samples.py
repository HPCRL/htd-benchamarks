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

mode = "tune"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_dilated_conv2d_nchw_{mode}.sh", 'w') as f:
    for item in arr:
        N, H, W, CO, CI, KH, KW, stride, pad, dilation = item
        output_height = (H + 2* pad - KH) // stride + 1
        output_width = (W + 2* pad - KW) // stride + 1
        with open("../../../test/dilated_conv2d_nchw/dilated_conv2d.mlir", "rt") as fin:
            with open(f"../../../test/dilated_conv2d_nchw/input_mlir/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_dilation_{dilation}_sm_80.mlir", "wt") as fout:
                # TODO: add dilation
                for line in fin:
                    fout.write(line.replace('${N}', str(N)).replace("${H}", str(H + 2 * pad)).replace("${W}", str(W + 2 * pad)).replace("${CO}", str(CO)).replace("${CI}", str(CI)).replace("${KH}", str(KW)).replace("${KW}", str(KW)).replace("${ST}", str(stride)).replace("${OH}", str(output_height)).replace("${OW}", str(output_width)))

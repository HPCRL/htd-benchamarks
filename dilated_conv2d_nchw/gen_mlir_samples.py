sizesVGG = [
    [1, 224, 224, 64, 3, 3, 3, 1, 1],   # VGG1
    [1, 112, 112, 128, 128, 3, 3, 1, 1], # VGG3
    [1, 56, 56, 256, 256, 3, 3, 1, 1],   # VGG5
    [1, 28, 28, 512, 512, 3, 3, 1, 1],   # VGG7
    [1, 14, 14, 512, 512, 3, 3, 1, 1],   # VGG9
]

sizesResnet = [
    [1, 224, 224, 64, 3, 7, 7, 2, 3],   # RESNET1
    [1, 56, 56, 64, 64, 1, 1, 1, 0],    # RESNET2
    [1, 56, 56, 64, 64, 3, 3, 1, 1],    # RESNET2
    [1, 56, 56, 256, 64, 1, 1, 1, 0],   # RESNET2
    [1, 56, 56, 128, 256, 1, 1, 2, 0],  # RESNET3
    [1, 28, 28, 128, 128, 3, 3, 1, 1],  # RESNET3
    [1, 28, 28, 512, 128, 1, 1, 1, 0],  # RESNET3
    [1, 28, 28, 256, 512, 1, 1, 2, 0],  # RESNET4
    [1, 14, 14, 256, 256, 3, 3, 1, 1],  # RESNET4
    [1, 14, 14, 1024, 256, 1, 1, 1, 0], # RESNET4
    [1, 14, 14, 512, 1024, 1, 1, 2, 0], # RESNET5
    [1, 7, 7, 512, 512, 3, 3, 1, 1],    # RESNET5
    [1, 7, 7, 2048, 512, 1, 1, 1, 0],   # RESNET5
]

sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo6
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo10
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 68, 68, 512, 256, 3, 3, 1, 1],   # yolo14
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo20
]
mode = "tune"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_dilated_conv2d_nchw_{mode}.sh", 'w') as f:
    for item in sizesVGG + sizesResnet + sizesYolo:
        N, H, W, CO, CI, KH, KW, stride, pad = item
        output_height = (H + 2* pad - KH) // stride + 1
        output_width = (W + 2* pad - KW) // stride + 1
        with open("../../../test/dilated_conv2d_nchw/dilated_conv2d.mlir", "rt") as fin:
            with open(f"../../../test/dilated_conv2d_nchw/input_mlir/dilated_conv2d_nchw_N{N}_H{H}_W{W}_CO{CO}_CI{CI}_KH{KH}_KW{KW}_ST_{stride}_padding_{pad}_sm_80.mlir", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('${N}', str(N)).replace("${H}", str(H + 2 * pad)).replace("${W}", str(W + 2 * pad)).replace("${CO}", str(CO)).replace("${CI}", str(CI)).replace("${KH}", str(KW)).replace("${KW}", str(KW)).replace("${ST}", str(stride)).replace("${OH}", str(output_height)).replace("${OW}", str(output_width)))

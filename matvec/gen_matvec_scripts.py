sizesMatvec = [
    [1024, 1024],   
    [1024, 512],
    [1024, 256],
    [1024, 128],
    [512, 512],
    [256, 256],
    [1000, 1000],
    [510, 510],
    [224, 224],
    [120, 120],
    [256, 1024],
    [224, 1024],
    [256, 1024],
    [224, 512],
    [120, 512],
    [1024, 32],
    [1024, 1024],
    [32, 1024],
]


mode = "tune"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_matvec_{mode}.sh", 'w') as f:
    for item in sizesMatvec:
        f.write(f"sbatch {mode}.slurm {' '.join(map(str, item))} {tune_number}\n")

mode = "eval"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_matvec_{mode}.sh", 'w') as f:
    for item in sizesMatvec:
        f.write(f"sbatch {mode}.slurm {' '.join(map(str, item))} {tune_number}\n")

mode = "resume"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_matvec_{mode}.sh", 'w') as f:
    for item in sizesMatvec:
        f.write(f"sbatch {mode}.slurm {' '.join(map(str, item))} {tune_number}\n")

mode = "timing"
tune_number = 1000 if mode == "tune" else ""
with open(f"all_matvec_{mode}.sh", 'w') as f:
    for item in sizesMatvec:
        N, K, M = item
        f.write(f'echo "Running matvec_M{N}_K{K}_N{M}"\n')
        # f.write(f"bash {mode}.slurm {' '.join(map(str, item))} &> ../../../results/matvec/tvm/matvec_M{N}_K{K}_N{M}_sm_80.txt\n")
        f.write(f"cd /uufs/chpc.utah.edu/common/home/u1419116/projects/hlt/codegen/tvm/src/matvec && module load cuda && cd ../../.. && cp auto_tuner_cpp/matvec/matvec_M{N}_K{K}_sm_80.cpp main.cpp && cd build && rm -rf * && cmake .. && make && ./codegen && cd ../test/mm/bash_scripts && bash matvec_M{M}_K{K}.sh &> ../../../results/matvec/autotuner/matvec_M{M}_K{K}_sm_80.txt && cd ../../../tvm/src/matvec/\n")
        f.write(f"cd /uufs/chpc.utah.edu/common/home/u1419116/projects/hlt/codegen/tvm/src/matvec && module load cuda && cd ../../.. && cp tvm/ansor_schedule/matvec_M{N}_K{K}_sm_80.cpp main.cpp && cd build && rm -rf * && cmake .. && make && ./codegen && cd ../test/mm/bash_scripts && bash matvec_M{M}_K{K}.sh &> ../../../results/matvec/same_schedule/matvec_M{M}_K{K}_sm_80.txt && cd ../../../tvm/src/matvec/\n")

for item in sizesMatvec:
    N, K = item
#     with open("../../../test/mm/matvec.mlir", "rt") as fin:
#         with open(f"../../../test/mm/input_mlir/matvec_M{N}_K{K}_N{M}.mlir", "wt") as fout:
#             for line in fin:
#                 fout.write(line.replace('${N}', str(N)).replace("${K}", str(K)).replace("${M}", str(M)))
                
    with open("../../../test/mm/test_mm.sh", "rt") as fin:
        with open(f"../../../test/mm/bash_scripts/matvec_M{N}_K{K}.sh", "wt") as fout:
            for line in fin:
                fout.write(line.replace('${N}', str(N)).replace("${K}", str(K)))
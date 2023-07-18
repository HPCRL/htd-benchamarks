sbatch eval.slurm 1 224 224 64 3 3 3 1 1 #VGG
sbatch eval.slurm 1 112 112 128 128 3 3 1 1 #VGG
sbatch eval.slurm 1 56 56 256 256 3 3 1 1 #VGG
sbatch eval.slurm 1 28 28 512 512 3 3 1 1 #VGG
sbatch eval.slurm 1 14 14 512 512 3 3 1 1 #VGG

sbatch eval.slurm 1 224 224 64 3 7 7 2 3 #RESNET
sbatch eval.slurm 1 56 56 64 64 1 1 1 0 #RESNET
sbatch eval.slurm 1 56 56 64 64 3 3 1 1 #RESNET
sbatch eval.slurm 1 56 56 256 64 1 1 1 0 #RESNET
sbatch eval.slurm 1 56 56 128 256 1 1 2 0 #RESNET
sbatch eval.slurm 1 28 28 128 128 3 3 1 1 #RESNET
sbatch eval.slurm 1 28 28 512 128 1 1 1 0 #RESNET
sbatch eval.slurm 1 28 28 256 512 1 1 2 0 #RESNET
sbatch eval.slurm 1 14 14 256 256 3 3 1 1 #RESNET
sbatch eval.slurm 1 14 14 1024 256 1 1 1 0 #RESNET
sbatch eval.slurm 1 14 14 512 1024 1 1 2 0 #RESNET
sbatch eval.slurm 1 7 7 512 512 3 3 1 1 #RESNET
sbatch eval.slurm 1 7 7 2048 512 1 1 1 0 #RESNET

sbatch eval.slurm 1 544 544 32 3 3 3 1 1 #YOLO
sbatch eval.slurm 1 272 272 64 32 3 3 1 1 #YOLO
sbatch eval.slurm 1 136 136 128 64 3 3 1 1 #YOLO
sbatch eval.slurm 1 136 136 64 128 1 1 1 0 #YOLO
sbatch eval.slurm 1 136 136 128 64 3 3 1 1 #YOLO
sbatch eval.slurm 1 68 68 256 128 3 3 1 1 #YOLO
sbatch eval.slurm 1 68 68 128 256 1 1 1 0 #YOLO
sbatch eval.slurm 1 68 68 256 128 3 3 1 1 #YOLO
sbatch eval.slurm 1 34 34 512 256 3 3 1 1 #YOLO
sbatch eval.slurm 1 34 34 256 512 1 1 1 0 #YOLO
sbatch eval.slurm 1 68 68 512 256 3 3 1 1 #YOLO
sbatch eval.slurm 1 17 17 1024 512 3 3 1 1 #YOLO
sbatch eval.slurm 1 17 17 512 1024 1 1 1 0 #YOLO
sbatch eval.slurm 1 17 17 1024 512 3 3 1 1 #YOLO

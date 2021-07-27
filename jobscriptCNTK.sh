#!/bin/bash
#SBATCH -p gpu20
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -t 50:00:00
#SBATCH -o example.log
#SBATCH --gres gpu:1


# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"

nvcc --version

python run_CNTKSketch_experiments.py --layers=3 --samples=5 --problem_type=class --filt_size=4 --first_logm=9 --last_logm=14 --tensor_degree=4 --lambdaa=0.01 --first_batch_size=400 --device_=cuda --input_dataset="./CIFAR/cifar.npz" --output=CIFAR.pdf

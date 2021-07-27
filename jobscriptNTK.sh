#!/bin/bash
#SBATCH -p cpu20
#SBATCH -c 48
#SBATCH --mem-per-cpu=4G
#SBATCH -t 50:00:00
#SBATCH -o example.log


python run_NTKSketch_experiments.py --layers=1 --samples=3 --problem_type=regress --first_logm=10 --last_logm=14 --tensor_degree=4 --lambdaa=0.3 --device_=cpu --input_dataset="./Location of CT slices/LocationCTData.npz" --output=CTLocation.pdf

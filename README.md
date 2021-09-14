# NTK-CNTK-Sketch
Implementation of the NTK and CNTK Sketch, fast methods for accelerating the learning with NTK and CNTK kernels

# Requirements
To compile and run tests you will need: quadprog package 

pip install quadprog

Furhthermore the Gradient Features experiments use autograd-hacks that can be cloned from the following repo:
https://github.com/cybertronai/autograd-hacks.git


# Making graphs
After installing the required packages and downloading the datasets and adding the path to the dataset, you can run the following command to run the experiments on NTK Sketch:

python run_NTKSketch_experiments.py --layers=1 --samples=3 --problem_type=regress --first_logm=10 --last_logm=14 --tensor_degree=4 --lambdaa=0.3 --device_=cpu --input_dataset="./path to the dataset/" --output=NTKGraph.pdf

Additionally, you can run the following command to run the experiments on CNTK Sketch:

python run_CNTKSketch_experiments.py --layers=3 --samples=5 --problem_type=class --filt_size=4 --first_logm=9 --last_logm=14 --tensor_degree=4 --lambdaa=0.01 --first_batch_size=400 --device_=cuda --input_dataset="./CIFAR/cifar.npz" --output=CIFARGraph.pdf

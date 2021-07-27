#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:48:05 2019

@author: zandieh
"""

import subprocess
import sys
import io
import argparse
import numpy as np
import torch
import time
import random
from numpy import linalg as npLA
import matplotlib.pyplot as plt
from CNTKSketchCuda import CNTKSketch, OblvFeatCNTK
from Grad_features import CNTKGradFeats


from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('x-large')
font.set_weight('book')


linestyles = ['-', '--', '-.', ':']
markers = ["p", "v", "D", "X"]
color_ = ['dodgerblue', 'orangered', 'mediumseagreen', 'gold']
ecolor_ = ['cornflowerblue', 'tomato', 'limegreen', 'yellow']

def make_graph_errobar(name, title, y_label, algs, algs_names, data, x_ticks, plot_scale):
    
    for i, alg in enumerate(algs):
        mean_ = np.mean(data[alg], axis=0)
        std_ = np.std(data[alg], axis=0)
        plt.errorbar(x_ticks[alg], mean_, yerr= std_, linestyle=linestyles[i], color= color_[i], linewidth=2.3, capsize=6, ecolor=ecolor_[i], label=algs_names[alg])
        

    plt.legend(prop=font, loc='best', frameon=True,fancybox=True,framealpha=0.8,edgecolor='k')
    plt.title(title, fontproperties = font)
    plt.xlabel("Feature Dimension", fontproperties = font)
    plt.ylabel(y_label, fontproperties = font)
    plt.grid()
    plt.yscale(plot_scale)
    plt.xscale('log', basex=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.savefig(name, dpi=500, bbox_inches='tight')
    plt.clf()


def make_graph(name, title, y_label, algs, algs_names, data, x_ticks, plot_scale):
    
    for i, alg in enumerate(algs):
        plt.plot(x_ticks[alg], data[alg], linestyle=linestyles[i], color= color_[i], linewidth=2.3, markersize=13, marker=markers[i], markerfacecolor=ecolor_[i], label=algs_names[alg])
        

    plt.legend(prop=font, loc='best', frameon=True,fancybox=True,framealpha=0.8,edgecolor='k')
    plt.title(title, fontproperties = font)
    plt.xlabel("Runtime (sec)", fontproperties = font)
    plt.ylabel(y_label, fontproperties = font)
    plt.grid()
    plt.yscale(plot_scale)
    plt.xscale('log', basex=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.savefig(name, dpi=500, bbox_inches='tight')
    plt.clf()



def CNTKSketch_KRR(L, filt_size, q, m, lambda_, Batch_size, device_, problem_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    n = Xtrain.shape[0]
    Channs = Xtest.shape[1]
    
    CNTK_Sketch = CNTKSketch(filt_size, Channs, m, q, L, device_)
    
    Features_train = np.zeros((n,m))
    Features_test = np.zeros((Xtest.shape[0],m))
    
    for epoch in range(int(n/Batch_size)):
        Features_train[epoch*Batch_size:(epoch+1)*Batch_size, :] = OblvFeatCNTK(CNTK_Sketch, torch.from_numpy(Xtrain[epoch*Batch_size:(epoch+1)*Batch_size, :, :, :]).to(CNTK_Sketch.device_)).to('cpu').data.numpy()
    
    for epoch in range(int(Xtest.shape[0]/Batch_size)):
        Features_test[epoch*Batch_size:(epoch+1)*Batch_size, :] = OblvFeatCNTK(CNTK_Sketch, torch.from_numpy(Xtest[epoch*Batch_size:(epoch+1)*Batch_size, :, :, :]).to(CNTK_Sketch.device_)).to('cpu').data.numpy()
    
    w_star = npLA.solve(np.dot(Features_train.T, Features_train)+lambda_*np.eye(Features_train.shape[1]), np.dot(Features_train.T, Ytrain))
    
    est = np.dot(Features_test, w_star)
    
    if problem_type == 'regress':
        
        TestErr = npLA.norm(est - Ytest)/np.sqrt(len(Ytest))
        
    elif problem_type == 'class':
        
        est = (est == est.max(axis=1, keepdims=True)).astype(int)-0.1
        TestErr = 100 - 100*npLA.norm(np.count_nonzero(est - Ytest, axis=1), 0)/len(Ytest)

    
    elapsed_time = time.time() - start_time
    
    return TestErr, elapsed_time



def Grad_Feat(L, filt_size, m, lambda_, Batch_size, device_, problem_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    n = Xtrain.shape[0]
    Channs = Xtest.shape[1]
    input_dim = Xtrain.shape[2]
    
    if L == 1:
        width = int(m/(Channs * filt_size**2) + 1)
    else:
        width = int(np.sqrt(max(1, m - np.sqrt(m/L)*Channs/filt_size)/(L-1))/filt_size + 1)
    
    grad_feat_cnn = CNTKGradFeats(filt_size, width, L, input_dim, Channs, Ytrain.shape[1], device_)
    
    m = grad_feat_cnn.num_features()
    
    Features_train = np.zeros((n,m))
    Features_test = np.zeros((Xtest.shape[0],m))
    
    for epoch in range(int(n/Batch_size)):
        Features_train[epoch*Batch_size:(epoch+1)*Batch_size, :] = grad_feat_cnn.ComputeFeats(Xtrain[epoch*Batch_size:(epoch+1)*Batch_size, :, :, :]).to('cpu').data.numpy()
    
    for epoch in range(int(Xtest.shape[0]/Batch_size)):
        Features_test[epoch*Batch_size:(epoch+1)*Batch_size, :] = grad_feat_cnn.ComputeFeats(Xtest[epoch*Batch_size:(epoch+1)*Batch_size, :, :, :]).to('cpu').data.numpy()

    
    w_star = npLA.solve(np.dot(Features_train.T, Features_train)+lambda_*np.eye(Features_train.shape[1]), np.dot(Features_train.T, Ytrain))
    
    est = np.dot(Features_test, w_star)
    
    if problem_type == 'regress':
        
        TestErr = npLA.norm(est - Ytest)/np.sqrt(len(Ytest))
        
    elif problem_type == 'class':
        
        est = (est == est.max(axis=1, keepdims=True)).astype(int)-0.1
        TestErr = 100 - 100*npLA.norm(np.count_nonzero(est - Ytest, axis=1), 0)/len(Ytest)

    elapsed_time = time.time() - start_time
    
    return TestErr, elapsed_time, m

    

def run_KRR(algs, args, Xtrain, Ytrain, Xtest, Ytest):
    
    start = args.first_logm
    end = args.last_logm
    
    runtimes = dict()
    test_errors = dict()
    feat_dim = dict()
    
    for alg_ in algs:
        
        runtimes[alg_] = np.zeros((args.samples, end - start + 1))
        test_errors[alg_] = np.zeros(runtimes[alg_].shape)
        feat_dim[alg_] = np.zeros(end - start + 1)
        
        for s in range(args.samples):
            for j, logm in enumerate(range(start, end+1)):
                
                m = 2**logm
                q = args.tensor_degree
                L = args.layers
                filt_size = args.filt_size
                Batch_size = int(args.first_batch_size * 2**start / m)
                
                if alg_ == 'CNTKSketch':
                    
                    test_errors[alg_][s,j], runtimes[alg_][s,j] = CNTKSketch_KRR(L, filt_size, q, m, args.lambdaa, Batch_size, args.device_, args.problem_type, Xtrain, Ytrain, Xtest, Ytest)
                    feat_dim[alg_][j] = m
                print(alg_, s, m, test_errors[alg_][s,j], runtimes[alg_][s,j])
    
    return runtimes, test_errors, feat_dim
    

def run_GradFeat(args, Xtrain, Ytrain, Xtest, Ytest):
    
    feat_dim = 2**np.linspace(args.first_logm, args.last_logm+1, num=args.last_logm - args.first_logm+3)
    actual_feat_dim = np.zeros(len(feat_dim))
    
    runtimes = np.zeros((args.samples, len(feat_dim)))
    test_errors = np.zeros(runtimes.shape)
    
    for j, m in enumerate(feat_dim):
        for s in range(args.samples):
            L = args.layers
            
            test_errors[s,j], runtimes[s,j], actual_feat_dim[j] = Grad_Feat(L, args.filt_size, m, args.lambdaa, args.first_batch_size, args.device_, args.problem_type, Xtrain, Ytrain, Xtest, Ytest)
            print('GradFeat', s, actual_feat_dim[j], test_errors[s,j], runtimes[s,j])
            
    return runtimes, test_errors, actual_feat_dim

def get_args():
    parser = argparse.ArgumentParser(description="Run regression and classification experiments with CNTKSketch.")
    parser.add_argument("--plot_scale", choices=['log', 'linear'], default="linear", help='the scale of y axis in the plots')
    parser.add_argument("--samples", action='store', default=3, type=int, help='number of samples to average over')
    parser.add_argument("--layers", action='store', default=3, type=int, help='number of layers of the NTK kernel')
    parser.add_argument("--filt_size", action='store', default=4, type=int, help='side-length of the convolutional filter')
    parser.add_argument("--first_logm", action='store', default=9, type=int, help='logarithm of the first value of features dimension')
    parser.add_argument("--last_logm", action='store', default=13, type=int, help='logarithm of the last value of features dimension')
    parser.add_argument("--first_batch_size", action='store', default=2000, type=int, help='the batch size for first logm')
    parser.add_argument("--problem_type", choices=['class', 'regress'], default='class', help='choose the problem type: classification or regression')
    parser.add_argument("--tensor_degree", choices=[2, 4, 8], default=4, type=int, help='the degree of the polynomial approximation to the arccosine functions')
    parser.add_argument("--device_", choices=['cuda', 'cpu'], default='cpu', help='choose the processing device')
    parser.add_argument("--lambdaa", action='store', default=0.01, type=float, help='regularizer in ridge regression')
    parser.add_argument("--input_dataset", action='store', default="./MNIST/mnistv.npz", type=str, help='location of the inpute dataset')
    parser.add_argument("--output", action='store', default="Graph.pdf", type=str, help='output graph name')

    args = parser.parse_args()
    
    return args


args = get_args() 

algs = [
    'CNTKSketch'
]


plot_names = {
    'CNTKSketch': 'CNTKSketch',
    'GradFeat' : 'GradientFeatures'
}

title = '{} with depth-{} CNTK, regularizer={}'.format(
        "Classification" if args.problem_type == 'class' else 'Regression', args.layers, args.lambdaa
    )


data = np.load(args.input_dataset)
Xtrain = data['arr_0']
Ytrain = data['arr_1']
Xtest = data['arr_2']
Ytest = data['arr_3']



runtime, test_error, feat_dim = run_KRR(algs, args, Xtrain, Ytrain, Xtest, Ytest)
# runtime = dict()
# test_error = dict()
# feat_dim = dict()


if args.problem_type == 'regress':
    
    ylabel = "Test RMSE"
        
elif args.problem_type == 'class':
    
    ylabel = "Test Accuracy (%)"



algs.append('GradFeat')
runtime['GradFeat'], test_error['GradFeat'], feat_dim['GradFeat'] = run_GradFeat(args, Xtrain, Ytrain, Xtest, Ytest)

make_graph_errobar('error-CNTK-GradFeat-' + args.output, title, ylabel, algs, plot_names, test_error, feat_dim, args.plot_scale)
make_graph_errobar('runtime-CNTK-GradFeat-' + args.output, title, "Runtime (sec)", algs, plot_names, runtime, feat_dim, args.plot_scale)


for alg in algs:
    test_error[alg] = np.mean(test_error[alg], axis=0)
    runtime[alg] = np.mean(runtime[alg], axis=0)

make_graph('runtime-' + args.output, title, ylabel, algs, plot_names, test_error, runtime, args.plot_scale)



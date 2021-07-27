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
from NTKSketchCuda import TensorSketch, OblvFeat, NTK
from NystromFeatures import NF
from sklearn.neural_network import MLPClassifier, MLPRegressor



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
    # ax = plt.axes()
    # ax.set_xticks(p)
    
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
    plt.xscale('log', base=2)
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
    plt.xscale('log', base=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.savefig(name, dpi=500, bbox_inches='tight')
    plt.clf()



def NTKSketch_KRR(L, q, m, lambda_, device_, problem_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    n = Xtrain.shape[0]
    d = Xtest.shape[1]
    
    Tensor_Sketch = TensorSketch(d, m, q, device_)
    
    NTK_coeff = NTK(L, q)
    
    Features_train = OblvFeat(Tensor_Sketch, torch.from_numpy(Xtrain).to(Tensor_Sketch.device_), NTK_coeff).to('cpu').data.numpy()
    Features_test = OblvFeat(Tensor_Sketch, torch.from_numpy(Xtest).to(Tensor_Sketch.device_), NTK_coeff).to('cpu').data.numpy()
    
    w_star = npLA.solve(np.dot(Features_train.T, Features_train)+lambda_*np.eye(Features_train.shape[1]), np.dot(Features_train.T, Ytrain))
    
    est = np.dot(Features_test, w_star)
    
    if problem_type == 'regress':
        
        TestErr = npLA.norm(est - Ytest)/np.sqrt(len(Ytest))
        
    elif problem_type == 'class':
        
        est = (est == est.max(axis=1, keepdims=True)).astype(int)-0.1
        TestErr = 100 - 100*npLA.norm(np.count_nonzero(est - Ytest, axis=1), 0)/len(Ytest)

    
    elapsed_time = time.time() - start_time
    
    return TestErr, elapsed_time

def NystromNTK(L, m, lambda_, problem_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    n = Xtrain.shape[0]
    d = Xtest.shape[1]
    
    Features_train, Features_test = NF(L, lambda_, m, Xtrain, Xtest)
    
    w_star = npLA.solve(np.dot(Features_train.T, Features_train)+lambda_*np.eye(Features_train.shape[1]), np.dot(Features_train.T, Ytrain))
    
    est = np.dot(Features_test, w_star)
    
    if problem_type == 'regress':
        
        TestErr = npLA.norm(est - Ytest)/np.sqrt(len(Ytest))
        
    elif problem_type == 'class':
        
        est = (est == est.max(axis=1, keepdims=True)).astype(int)-0.1
        TestErr = 100 - 100*npLA.norm(np.count_nonzero(est - Ytest, axis=1), 0)/len(Ytest)

    
    elapsed_time = time.time() - start_time
    
    return TestErr, elapsed_time


def MLP(L, m, lambda_, problem_type, Xtrain, Ytrain, Xtest, Ytest):
    start_time = time.time()
    n = Xtrain.shape[0]
    d = Xtest.shape[1]
    
    if problem_type == 'regress':
        
        if L == 1:
            width = int(m/(d+1)+1)
        else:
            width = int(np.sqrt(max(1, m - np.sqrt(m/L)*d)/(L-1))+1)
        
        model = MLPRegressor(hidden_layer_sizes=tuple([width]*L), alpha=lambda_, solver='sgd').fit(Xtrain, Ytrain)
        est = model.predict(Xtest)

        TestErr = npLA.norm(est - Ytest)/np.sqrt(len(Ytest))
        
    elif problem_type == 'class':
        
        if L == 1:
            width = int(m/(d+Ytrain.shape[1]) + 1)
        else:
            width = int(np.sqrt(max(1, m - np.sqrt(m/L)*d)/(L-1))+1)
        
        model = MLPRegressor(hidden_layer_sizes=tuple([width]*L), alpha=lambda_, solver='sgd').fit(Xtrain, Ytrain)
        est = model.predict(Xtest)
        est = (est == est.max(axis=1, keepdims=True)).astype(int)-0.1
        TestErr = 100 - 100*npLA.norm(np.count_nonzero(est - Ytest, axis=1), 0)/len(Ytest)

    elapsed_time = time.time() - start_time
    
    # weights = model.coefs_
    # biases = model.intercepts_
    # num_weights = 0
    # for i in range(len(weights)):
    #     num_weights += weights[i].shape[0]*weights[i].shape[1] + len(biases)
    
    return TestErr, elapsed_time#, num_weights

    

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
                
                if alg_ == 'NTKSketch':
                    
                    test_errors[alg_][s,j], runtimes[alg_][s,j] = NTKSketch_KRR(L, q, m, args.lambdaa, args.device_, args.problem_type, Xtrain, Ytrain, Xtest, Ytest)
                    feat_dim[alg_][j] = m
                    
                elif alg_ == 'Nystrom':
                    
                    test_errors[alg_][s,j], runtimes[alg_][s,j] = NystromNTK(L, m, args.lambdaa, args.problem_type, Xtrain, Ytrain, Xtest, Ytest)
                    feat_dim[alg_][j] = m
                
    
    return runtimes, test_errors, feat_dim
    

def run_MLP(args, Xtrain, Ytrain, Xtest, Ytest):
    
    feat_dim = 10 * 2**np.linspace(args.first_logm, args.last_logm, num=args.last_logm - args.first_logm+3)
    
    runtimes = np.zeros((args.samples, len(feat_dim)))
    test_errors = np.zeros(runtimes.shape)
    
    for j, m in enumerate(feat_dim):
        for s in range(args.samples):
            L = args.layers
            
            test_errors[s,j], runtimes[s,j] = MLP(L, m, args.lambdaa, args.problem_type, Xtrain, Ytrain, Xtest, Ytest)
    
    return runtimes, test_errors

def get_args():
    parser = argparse.ArgumentParser(description="Run regression and classification experiments with NTKSketch.")
    parser.add_argument("--plot_scale", choices=['log', 'linear'], default="linear", help='the scale of y axis in the plots')
    parser.add_argument("--samples", action='store', default=2, type=int, help='number of samples to average over')
    parser.add_argument("--layers", action='store', default=1, type=int, help='number of layers of the NTK kernel')
    parser.add_argument("--first_logm", action='store', default=9, type=int, help='logarithm of the first value of features dimension')
    parser.add_argument("--last_logm", action='store', default=11, type=int, help='logarithm of the last value of features dimension')
    parser.add_argument("--problem_type", choices=['class', 'regress'], default='class', help='choose the problem type: classification or regression')
    parser.add_argument("--tensor_degree", choices=[2, 4, 8], default=4, type=int, help='the degree of the polynomial approximation to the arccosine functions')
    parser.add_argument("--device_", choices=['cuda', 'cpu'], default='cpu', help='choose the processing device')
    parser.add_argument("--lambdaa", action='store', default=0.1, type=float, help='regularizer in ridge regression')
    parser.add_argument("--input_dataset", action='store', default="./MNIST/mnistv.npz", type=str, help='location of the inpute dataset')
    parser.add_argument("--output", action='store', default="Graph.pdf", type=str, help='output graph name')

    args = parser.parse_args()
    
    return args


args = get_args() 

algs = [
    'NTKSketch',
    'Nystrom'
]


plot_names = {
    'NTKSketch': 'NTKSketch ',
    'Nystrom' : 'Nystrom',
    'MLP' : 'SGD-trained MLP'
}

title = '{} with depth-{} NTK, regularizer={}'.format(
        "Classification" if args.problem_type == 'class' else 'Regression', args.layers, args.lambdaa
    )


data = np.load(args.input_dataset)
Xtrain = data['arr_0']
Ytrain = data['arr_1']
Xtest = data['arr_2']
Ytest = data['arr_3']



runtime, test_error, feat_dim = run_KRR(algs, args, Xtrain, Ytrain, Xtest, Ytest)


if args.problem_type == 'regress':
    
    ylabel = "Test RMSE"
        
elif args.problem_type == 'class':
    
    ylabel = "Test Accuracy (%)"


make_graph_errobar('error-NTK-Nystrom-' + args.output, title, ylabel, algs, plot_names, test_error, feat_dim, args.plot_scale)
make_graph_errobar('runtime-NTK-Nystrom-' + args.output, title, "Runtime (sec)", algs, plot_names, runtime, feat_dim, args.plot_scale)


algs.append('MLP')
runtime['MLP'], test_error['MLP'] = run_MLP(args, Xtrain, Ytrain, Xtest, Ytest)

for alg in algs:
    test_error[alg] = np.mean(test_error[alg], axis=0)
    runtime[alg] = np.mean(runtime[alg], axis=0)

make_graph('runtime-' + args.output, title, ylabel, algs, plot_names, test_error, runtime, args.plot_scale)



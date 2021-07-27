#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:48:05 2019

@author: zandieh
"""


import torch
import numpy as np
from numpy import linalg as LA
import copy

import sys
sys.path.append('./autograd-hacks')

import autograd_hacks

class CNN(torch.nn.Module):
    def __init__(self, input_channel, input_dim, filtersize, num_cv_layers, channels, outputsize):
        super(CNN, self).__init__()
        self.input_channel = input_channel
        self.filtersize = filtersize
        self.channels = channels
        self.feature_size = input_dim - num_cv_layers*(filtersize-1)
        self.num_cv_layers = num_cv_layers
        self.relu = torch.nn.ReLU()
        if self.num_cv_layers >= 1:
            self.conv1 = torch.nn.Conv2d(self.input_channel, self.channels, self.filtersize)
        if self.num_cv_layers >= 2:
            self.conv2 = torch.nn.Conv2d(self.channels, self.channels, self.filtersize)
        if self.num_cv_layers >= 3:
            self.conv3 = torch.nn.Conv2d(self.channels, self.channels, self.filtersize)
        if self.num_cv_layers >= 4:
            self.conv4 = torch.nn.Conv2d(self.channels, self.channels, self.filtersize)
        if self.num_cv_layers >= 5:
            self.conv5 = torch.nn.Conv2d(self.channels, self.channels, self.filtersize)
        self.GAP = torch.nn.AvgPool2d((self.feature_size, self.feature_size))
        self.fc = torch.nn.Linear(self.channels, outputsize) 
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=1.0/np.sqrt(self.fc.weight.size(1)))
        
    def forward(self, x):
        if self.num_cv_layers == 1:
            layer1 = self.relu(self.conv1(x))
            layer_gap = self.GAP(layer1)
            output = self.fc(layer_gap.view(-1, self.channels))
            return output
        elif self.num_cv_layers == 2:
            layer1 = self.relu(self.conv1(x))
            layer2 = self.relu(self.conv2(layer1))
            layer_gap = self.GAP(layer2)
            output = self.fc(layer_gap.view(-1, self.channels))
            return output
        elif self.num_cv_layers == 3:
            layer1 = self.relu(self.conv1(x))
            layer2 = self.relu(self.conv2(layer1))
            layer3 = self.relu(self.conv3(layer2))
            layer_gap = self.GAP(layer3)
            output = self.fc(layer_gap.view(-1, self.channels))
            return output
        elif self.num_cv_layers == 4:
            layer1 = self.relu(self.conv1(x))
            layer2 = self.relu(self.conv2(layer1))
            layer3 = self.relu(self.conv3(layer2))
            layer4 = self.relu(self.conv4(layer3))
            layer_gap = self.GAP(layer4)
            output = self.fc(layer_gap.view(-1, self.channels))
            return output
        elif self.num_cv_layers == 5:
            layer1 = self.relu(self.conv1(x))
            layer2 = self.relu(self.conv2(layer1))
            layer3 = self.relu(self.conv3(layer2))
            layer4 = self.relu(self.conv4(layer3))
            layer5 = self.relu(self.conv5(layer4))
            layer_gap = self.GAP(layer5)
            output = self.fc(layer_gap.view(-1, self.channels))
            return output
    


class CNTKGradFeats:
    def __init__(self, filt_size, width, num_layers, input_dim, input_channs, outputsize, dev):
        self.filt_size = filt_size
        self.width = width
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.channs = input_channs
        self.outputsize = outputsize
        self.device_ = dev
        self.cnn = CNN(self.channs, self.input_dim, self.filt_size , self.num_layers, self.width, self.outputsize).to(self.device_)


    def ComputeFeats(self, X):
        
        X = torch.FloatTensor(X.astype(np.float32)).to(self.device_)
        
        clone_cnn = copy.deepcopy(self.cnn).to(self.device_)
        autograd_hacks.add_hooks(clone_cnn)
        out = clone_cnn(X)      
        
        clone_cnn.zero_grad()
        out.backward(torch.ones(X.shape[0], self.outputsize, device=self.device_), retain_graph=True)
        autograd_hacks.compute_grad1(clone_cnn)
        autograd_hacks.disable_hooks()
        
        params = list(clone_cnn.parameters())
        
        grad_feats = []
        for i in range(len(params)):
            grad_feats.append(torch.flatten(params[i].grad1, start_dim=1))
        grad_feats = torch.hstack(grad_feats)
        
        autograd_hacks.clear_backprops(self.cnn)
        
        return grad_feats


    def num_features(self):
        num_params = self.channs * self.width * self.filt_size**2 + self.width
        num_params += (self.num_layers-1) * (self.width**2 * self.filt_size**2 + self.width)
        num_params += self.outputsize * self.width + self.outputsize
        
        return num_params
            
        
        
        
        

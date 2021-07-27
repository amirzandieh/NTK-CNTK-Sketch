#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:48:05 2019

@author: zandieh
"""
import numpy as np
from numpy import linalg as LA


def K_relu(L):
    Y = np.zeros((8192,L+1))
    Y[:,0] = np.linspace(-1.0, 1.0, num=8192)
    for i in range(L):
        Y[:,i+1] = (np.sqrt(1-Y[:,i]**2) + Y[:,i]*(np.pi - np.arccos(Y[:,i])))/np.pi
    
    y = np.zeros(8192)
    for i in range(L+1):
        z = Y[:,i]
        for j in range(i,L):
            z = z*(np.pi - np.arccos(Y[:,j]))/np.pi 
        y = y + z
        
    return y

def RecNS(NTK_, lambda_, m, Data):
    
    n = Data.shape[0]
    
    if m >= n:
        S = np.arange(n)
        W = np.ones(n)
        return S, W
    
    
    D = Data[np.random.choice(n, int(n/2), replace=False),:]
    S1, W1 = RecNS(NTK_, lambda_, m, D)
    X = D[S1,:]
    
    
    Normalizer = LA.norm(X, axis=1)
    Xnormalized = np.divide(X.T, Normalizer, out=np.zeros_like(X.T), where=Normalizer!=0).T
    
    Gram = ((len(NTK_)-1)/2)*(1+np.dot(Xnormalized, Xnormalized.T))
    Gram = Gram.astype(np.uint16)

    
    KSS = np.take(NTK_, Gram)
    KSS = (KSS * Normalizer).T * Normalizer
    Gram = 0

    KSS = (KSS*W1).T*W1
    
    
    Norm_data = LA.norm(Data, axis=1)
    DataNormalized = np.divide(Data.T, Norm_data, out=np.zeros_like(Data.T), where=Norm_data!=0).T
    
    Gram_data = ((len(NTK_)-1)/2)*(1+np.dot(DataNormalized, Xnormalized.T))
    Gram_data = Gram_data.astype(np.uint16)
    
    KS = np.take(NTK_, Gram_data)
    KS = ((KS * Normalizer).T * Norm_data).T
    Gram_data = 0
    
    KS = KS*W1
    
    
    M = LA.solve(KSS+lambda_*np.eye(KSS.shape[0]), KS.T)

    L = (NTK_[-1])*Norm_data**2 - np.einsum("ij,ij->j", KS.T, M)

        
    L[L<0] = 0
    p = L/np.sum(L)
    S = np.random.choice(n, m, replace=False, p=p)
    W = 1/np.sqrt(m*p[S])
    
    return S, W


def NF(L, lambda_, m, Xtrain, Xtest):
    
    NTK_ = K_relu(L)
    
    S = RecNS(NTK_, lambda_, m, Xtrain)[0]
    C = Xtrain[S,:]
    
    
    Normalizer_C = LA.norm(C, axis=1)
    Cnormalized = np.divide(C.T, Normalizer_C, out=np.zeros_like(C.T), where=Normalizer_C!=0).T
    
    Gram_C = ((len(NTK_)-1)/2)*(1+np.dot(Cnormalized, Cnormalized.T))
    Gram_C = Gram_C.astype(np.uint16)

    K_C = np.take(NTK_, Gram_C)
    
    P, D, Q = LA.svd(K_C, full_matrices=False)
    D = np.sqrt(D)
    D[D<0.000001] = 0
    Rotation = Q.T * np.divide(1, D, out=np.zeros_like(D), where=D!=0)
    
    
    Norm_trn = LA.norm(Xtrain, axis=1)
    X_tr_Norm = np.divide(Xtrain.T, Norm_trn, out=np.zeros_like(Xtrain.T), where=Norm_trn!=0).T
    
    Gram_trn = ((len(NTK_)-1)/2)*(1+np.dot(X_tr_Norm, Cnormalized.T))
    Gram_trn = Gram_trn.astype(np.uint16)
    
    Feat_trn = np.take(NTK_, Gram_trn)
    Feat_trn = (np.dot(Feat_trn, Rotation).T * Norm_trn).T

    

    Norm_tst = LA.norm(Xtest, axis=1)
    X_tst_Norm = np.divide(Xtest.T, Norm_tst, out=np.zeros_like(Xtest.T), where=Norm_tst!=0).T
    
    Gram_tst = ((len(NTK_)-1)/2)*(1+np.dot(X_tst_Norm, Cnormalized.T))
    Gram_tst = Gram_tst.astype(np.uint16)
    
    Feat_tst = np.take(NTK_, Gram_tst)
    Feat_tst = (np.dot(Feat_tst, Rotation).T * Norm_tst).T

    return Feat_trn, Feat_tst



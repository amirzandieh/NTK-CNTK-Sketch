#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:48:05 2019

@author: zandieh
"""
import numpy as np
import torch
import torch.fft
from torch import linalg as LA
import math
import quadprog
import time



def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T +0.00001*np.eye(P.shape[0]))   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def NTK(L,q):
    n=15*L+5*q
    Y = np.zeros((201+n,L+1))
    Y[:,0] = np.sort(np.concatenate((np.linspace(-1.0, 1.0, num=201), np.cos((2*np.arange(n)+1)*np.pi / (4*n))), axis=0))

    m_ = Y.shape[0]

    for i in range(L):
        Y[:,i+1] = (np.sqrt(1-Y[:,i]**2) + Y[:,i]*(np.pi - np.arccos(Y[:,i])))/np.pi
    
    y = np.zeros(m_)
    for i in range(L+1):
        z = Y[:,i]
        for j in range(i,L):
            z = z*(np.pi - np.arccos(Y[:,j]))/np.pi
        y = y + z
    
    Z = np.zeros((m_,q+1))
    Z[:,0] = np.ones(m_)
    for i in range(q):
        Z[:,i+1] = Z[:,i] * Y[:,0]
    
    
    weight_ = np.linspace(0.0, 1.0, num=m_) + 2/L
    w = y * weight_
    U = Z.T * weight_

    beta_ = quadprog_solve_qp(np.dot(U, U.T), -np.dot(U,w) , np.concatenate((Z[0:m_-1,:]-Z[1:m_,:], -np.eye(q+1)),axis=0), np.zeros(q+m_))
    beta_[beta_ < 0.00001] = 0
    
    return beta_

def TSRHTCmplx(X1, X2, P, D):
    
    Xhat1 = torch.fft.fftn(X1 * D[0,:], dim=1)[:,P[0,:]]
    Xhat2 = torch.fft.fftn(X2 * D[1,:], dim=1)[:,P[1,:]]
    
    Y = np.sqrt(1/P.shape[1])*(Xhat1 * Xhat2)
    
    return Y
    
def SRHTCmplx_Stndrd(X, P, D):
    return np.sqrt(1/len(P)) * torch.fft.fftn(X * D, dim=1)[:,P]

class TensorSketch:
    def __init__(self, d, m, q, dev):
        self.d = d
        self.m = m
        self.q = q
        self.device_ = dev
    
        self.Tree_D = [0 for i in range((self.q-1).bit_length())]
        self.Tree_P = [0 for i in range((self.q-1).bit_length())]
    
        m_=int(self.m/4)
        q_ = int(self.q/2)
        for i in range((self.q-1).bit_length()):
            if i == 0:
                self.Tree_P[i] = torch.from_numpy(np.random.choice(self.d, (q_,2,m_))).to(self.device_)
                self.Tree_D[i] = torch.from_numpy(np.random.choice((-1,1), (q_,2,self.d))).to(self.device_)
            else:
                self.Tree_P[i] = torch.from_numpy(np.random.choice(m_, (q_,2,m_))).to(self.device_)
                self.Tree_D[i] = torch.from_numpy(np.random.choice((-1,1), (q_,2,m_))).to(self.device_)
            q_ = int(q_/2)
        
        self.D = torch.from_numpy(np.random.choice((-1,1), self.q*m_)).to(self.device_)
        self.P = torch.from_numpy(np.random.choice(self.q*m_, int(self.m/2-1))).to(self.device_)
            

    def Sketch(self, X):
        n=X.shape[0]
        lgq = len(self.Tree_D)
        V = [0 for i in range(lgq)]
        #E1 = torch.cat((torch.ones((n,1), device=self.device_), torch.zeros((n,X.shape[1]-1), device=self.device_)), 1)
        
        for i in range(lgq):
            q = self.Tree_D[i].shape[0]
            V[i] = torch.zeros((q,n,self.Tree_P[i].shape[2]), dtype=torch.cfloat, device=self.device_)
            for j in range(q):
                if i == 0:
                    V[i][j,:,:] = TSRHTCmplx(X, X, self.Tree_P[i][j,:,:], self.Tree_D[i][j,:,:])
                else:
                    V[i][j,:,:] = TSRHTCmplx(V[i-1][2*j,:,:], V[i-1][2*j+1,:,:], self.Tree_P[i][j,:,:], self.Tree_D[i][j,:,:])
    
        U = [0 for i in range(2**lgq)]
        U[0] = V[lgq-1][0,:,:].detach().clone()
        
        SetE1 = {}
        
        for j in range(1,len(U)):
            p = int((j-1)/2)
            for i in range(lgq):
                if j%(2**(i+1)) == 0 :
                    SetE1[(i,p)] = 1
                    #V[i][p,:,:] = torch.cat((torch.ones((n,1)), torch.zeros((n,V[i].shape[2]-1))), 1)
                else:
                    if i == 0:
                        V[i][p,:,:] = SRHTCmplx_Stndrd(X, self.Tree_P[i][p,0,:], self.Tree_D[i][p,0,:])
                    else:
                        if (i-1,2*p) in SetE1:
                            V[i][p,:,:] = V[i-1][2*p+1,:,:]
                        else:
                            V[i][p,:,:] = TSRHTCmplx(V[i-1][2*p,:,:], V[i-1][2*p+1,:,:], self.Tree_P[i][p,:,:], self.Tree_D[i][p,:,:])
                p = int(p/2)
            U[j] = V[lgq-1][0,:,:].detach().clone()
        
        return U


def OblvFeat(tensr_sktch, X, coeff):
    q = tensr_sktch.q
    n = X.shape[0]
    norm_X = LA.norm(X, dim=1)
    Normalizer = torch.where(norm_X>0, norm_X, 1.0)
    Xnormalized = ((X.T / Normalizer).T)
    U = tensr_sktch.Sketch(Xnormalized)
    m = U[0].shape[1]
    
    Z = torch.zeros((len(tensr_sktch.D),n), dtype=torch.cfloat, device=tensr_sktch.device_)
    for i in range(q):
        Z[m*i:m*(i+1)] = np.sqrt(coeff[i+1]) * U[q-i-1].T
        U[q-i-1]=0
    
    Z = (np.sqrt(1/len(tensr_sktch.P))*torch.fft.fftn(Z.T*tensr_sktch.D, dim=1)[:,tensr_sktch.P])
    Z = (Z.T * Normalizer).T
    return torch.cat((np.sqrt(coeff[0]) * Normalizer.reshape((n,1)), torch.cat((Z.real, Z.imag), 1)), 1)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:00:27 2018
Functions to perform multivariate U-test
@author: antoine
"""
import skcuda
from skcuda import linalg as cuda_linalg
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
cuda_linalg.init()
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
import matplotlib.pyplot as plt


import numpy as np
def random_vector(size):
    """  generate a random vector
    size: size of the vector
    """
    vect=np.random.normal(size=size)
    return vect


from scipy.stats import mannwhitneyu
def random_projection_u_test(x,y,alpha=0.05):
    """  Multivariate U-test using random projection
    x,y: sample to compare
    alpha: significance level
    """
    vect=random_vector(y.shape[1])
    proj_x=vect.dot(x.transpose())
    proj_y=vect.dot(y.transpose())
    
    test=mannwhitneyu(proj_x,proj_y,alternative='greater')
    return test[1]<alpha



def kPC_U_test(x,y,kernel_width=1,alpha=0.05):
    """  kernel PCA multivariate U-test
    x,y: sample to compare
    kernel_width:width of the kernel
    alpha: significance level
    """
    ones_N=np.ones((x.shape[0],x.shape[0]))
    ##Gram matrice for X
    gram_x=rbf_kernel(x,x,1/(2*kernel_width**2))
    ##Center Gram matrice
    gram_x=gram_x-ones_N.dot(gram_x)-gram_x.dot(ones_N)+ones_N.dot(gram_x).dot(ones_N)
    ##Gram matrice for y
    gram_y=rbf_kernel(y,y,1/(2*kernel_width**2))
    ##Center Gram matrice
    gram_y=gram_y-ones_N.dot(gram_y)-gram_y.dot(ones_N)+ones_N.dot(gram_y).dot(ones_N)
    ##Eigenvalues computation
    pc_x=np.linalg.eigvalsh(gram_x)
    pc_y=np.linalg.eigvalsh(gram_y)
    ##U_test
    U_test=mannwhitneyu(pc_x,pc_y,alternative='two-sided')
    return U_test[1]<alpha


def U_test_gpu(gram_x,gram_y,alpha=0.05):
    """  kernel PCA multivariate U-test (GPU implementation)
    x,y: sample to compare
    kernel_width:width of the kernel
    alpha: significance level
    """
    ##Gram matrice for X
    gram_x=gpuarray.to_gpu(gram_x)
    ##Center Gram matrice
    ##X eigenvalues computation
    _, w_gpu = cuda_linalg.eig(gram_x, lib= 'cusolver')
    pc_x=w_gpu.get()
    
    ##Gram matrice for Y
    gram_y=gpuarray.to_gpu(gram_y)
    ##Center Gram matrice
    ##Y eigenvalues computation
    eigen, w_gpu = cuda_linalg.eig(gram_y, lib='cusolver')
    pc_y=w_gpu.get()
    ##U_test
    U_test=mannwhitneyu(pc_x,pc_y,alternative='two-sided')
    print(U_test)
    return U_test[1]<alpha, eigen.get()

def U_test_gpu_large(gram_x,gram_y,n_eigen_value=256):
    """  kernel PCA multivariate U-test (GPU implementation)
    x,y: sample to compare
    kernel_width:width of the kernel
    alpha: significance level
    """
    n_eigen_value=min(n_eigen_value,gram_x.shape[0])
    n_rep=max(10,int(gram_x.shape[0]/n_eigen_value)*2)
    eigenvalue_x=np.zeros(n_eigen_value*n_rep)
    eigenvalue_y=np.zeros(n_eigen_value*n_rep)
    for repetition in range(n_rep):
        idx = list(np.random.randint(gram_x.shape[0], size=n_eigen_value))
        ##Gram matrice for X
        gram_x_gpu=gpuarray.to_gpu(gram_x[idx,:][:,idx]/n_eigen_value)
        ##Center Gram matrice
        ##X eigenvalues computation
        _,w_gpu = cuda_linalg.eig(gram_x_gpu, lib= 'cusolver')
        pc_x=w_gpu.get()
        eigenvalue_x[repetition*n_eigen_value:(repetition+1)*n_eigen_value]=pc_x

    for repetition in range(n_rep):
        ##Gram matrice for Y
        idy = list(np.random.randint(gram_y.shape[0], size=n_eigen_value))
        gram_y_gpu=gpuarray.to_gpu(gram_y[idy,:][:,idy]/n_eigen_value)
        ##Center Gram matrice
        ##Y eigenvalues computation
        _, w_gpu = cuda_linalg.eig(gram_y_gpu, lib='cusolver')
        pc_y=w_gpu.get()
        eigenvalue_y[repetition*n_eigen_value:(repetition+1)*n_eigen_value]=pc_y
        ##U_test
    U_test=mannwhitneyu(eigenvalue_x,eigenvalue_y,alternative='two-sided')
    eigen_x_spike=eigenvalue_x[eigenvalue_x>np.percentile(eigenvalue_x,95)]
    eigen_y_spike=eigenvalue_y[eigenvalue_y>np.percentile(eigenvalue_y,95)]

    U_test_spike=mannwhitneyu(eigen_x_spike,eigen_y_spike,alternative='two-sided')
    return U_test[1],U_test_spike[1]


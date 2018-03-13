#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:50:15 2018

@author: antoine
"""

from skcuda import linalg as cuda_linalg
import pycuda.gpuarray as gpuarray
cuda_linalg.init()


def eigh_gpu(X):
    X=gpuarray.to_gpu(X)
    eigen, w_gpu = cuda_linalg.eig(X, lib='cusolver')
    return eigen.get(),w_gpu.get()
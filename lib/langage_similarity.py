#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:51:25 2018

@author: antoine
"""
from .w2vec import Word2vec
from .build_graph import build_graph
from .mutivariate_u_test import U_test_gpu_large
import numpy as np
from .eigh_gpu import eigh_gpu
from scipy.stats import mannwhitneyu


def U_test_langage_large(lang_A,lang_B,dict_size=4000,random=1):
    graph_A=build_langage_graph(lang_A,dict_size,random)
    graph_B=build_langage_graph(lang_B,dict_size,random)
    return U_test_gpu_large(graph_A,graph_B)


def build_langage_graph(lang,dict_size,random=1):
    file='fastText/wiki.'+lang+'.vec'
    w2vect=Word2vec(file,dict_size,random)
    sim=w2vect.vocab_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return build_graph(sim)

def build_langage_semantic_graph(lang,dict_size,random=1):
    file='fastText/wiki.'+lang+'.vec'
    w2vect=Word2vec(file,dict_size,random)
    sim=w2vect.semantic_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return build_graph(sim)

def build_langage_semantic_sim(lang,dict_size,random=1):
    file='fastText/wiki.'+lang+'.vec'
    w2vect=Word2vec(file,dict_size,random)
    sim=w2vect.semantic_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return sim
    

def U_test_langage(lang_A,lang_B,dict_size=4000,random=1):
    graph_A=build_langage_graph(lang_A,dict_size,random)
    graph_A*=1/graph_A.shape[0]
    
    graph_B=build_langage_graph(lang_B,dict_size,random)
    graph_B*=1/graph_B.shape[0]
    
    eigvect_A,eigval_A=eigh_gpu(graph_A)
    eigvect_B,eigval_B=eigh_gpu(graph_B)
    eig_spike_A=eigval_A[eigval_A>np.percentile(eigval_A,95)]
    eig_spike_B=eigval_B[eigval_B>np.percentile(eigval_B,95)]
    
    U_test_spike=mannwhitneyu(eig_spike_A,eig_spike_B,alternative='two-sided')[1]
    U_test=mannwhitneyu(eigval_A,eigval_B,alternative='two-sided')[1]

    return U_test,U_test_spike

import matplotlib.pyplot as plt
def U_test_langage_semantic(lang_A,lang_B,dict_size=100000,random=1,n_graph=100,plot_spectrum=False):
    sim_A=build_langage_semantic_sim(lang_A,dict_size,random)
    sim_B=build_langage_semantic_sim(lang_B,dict_size,random)
    n_eigen=sim_A.shape[0]
    
    eigvals_A=np.zeros(n_eigen*n_graph)
    eigvals_B=np.zeros(n_eigen*n_graph)
    for i in range(n_graph):
        graph_A=build_graph(sim_A)
        graph_A=(graph_A-np.mean(graph_A))/np.std(graph_A)
        graph_B=build_graph(sim_B)
        graph_B=(graph_B-np.mean(graph_B))/np.std(graph_B)
        eigvect_A,eigval_A=eigh_gpu(graph_A/np.sqrt(graph_A.shape[0]))
        eigvect_B,eigval_B=eigh_gpu(graph_B/np.sqrt(graph_B.shape[0]))
        eigvals_A[i*n_eigen:(i+1)*n_eigen]=eigval_A
        eigvals_B[i*n_eigen:(i+1)*n_eigen]=eigval_B
    eig_spike_A=eigvals_A[eigvals_A>2+np.mean(eigvals_A)]
    eig_spike_B=eigvals_B[eigvals_B>2+np.mean(eigvals_B)]
    if plot_spectrum:
        plt.hist(eigval_A,alpha=0.5,bins=100,density=True)
        plt.hist(eigval_B,alpha=0.5,bins=100,density=True)
        plt.title("Spectrum")
        plt.show()
        
        plt.hist(eig_spike_A,alpha=0.5,bins=50,density=True)
        plt.hist(eig_spike_B,alpha=0.5,bins=50,density=True)
        plt.title("Spike Spectrum")
        plt.show()
    print("# Spike A:",len(eig_spike_A))
    print("# Spike B:",len(eig_spike_B))
    U_test_spike=mannwhitneyu(eig_spike_A,eig_spike_B,alternative='two-sided')[1]
    U_test=mannwhitneyu(eigvals_A,eigvals_B,alternative='two-sided')[1]
    return U_test,U_test_spike
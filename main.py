#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:31:19 2018

@author: antoine
"""

from lib.w2vec import Word2vec
import numpy as np

def build_graph(sim_matrix):
    random_mat=np.random.uniform(size=sim_matrix.shape)
    random_mat=np.tril(random_mat)+np.transpose(np.tril(random_mat))
    return (1-sim_matrix <= random_mat).astype(float)

from lib.mutivariate_u_test import U_test_gpu,U_test_gpu_large
DIM=50000
w2vect_fr=Word2vec('fastText/wiki.fr.vec',DIM)
sem_sim_fr=w2vect_fr.semantic_similarity()
sem_sim_fr=(sem_sim_fr-np.min(sem_sim_fr))/(np.max(sem_sim_fr)-np.min(sem_sim_fr))

w2vect_en=Word2vec('fastText/wiki.en.vec',DIM)
sem_sim_en=w2vect_en.semantic_similarity()
sem_sim_en=(sem_sim_en-np.min(sem_sim_en))/(np.max(sem_sim_en)-np.min(sem_sim_en))



w2vect_en=Word2vec('fastText/wiki.ro.vec',DIM)
sim_fr=w2vect_fr.vocab_similarity()
sim_fr=(sim_fr-np.min(sim_fr))/(np.max(sim_fr)-np.min(sim_fr))
sim_en=w2vect_en.vocab_similarity()
sim_en=(sim_en-np.min(sim_en))/(np.max(sim_en)-np.min(sim_en))

from lib.eigh_gpu import eigh_gpu
graph_fr=build_graph(sim_fr)/np.sqrt(DIM)
eigvect_fr,eigval_fr=eigh_gpu(graph_fr)
graph_en=build_graph(sim_en)/np.sqrt(DIM)
eigvect_en,eigval_en=eigh_gpu(graph_en)

from scipy.stats import mannwhitneyu
eig_big_en=eigval_en[eigval_en>np.percentile(eigval_en,95)]
eig_big_fr=eigval_fr[eigval_fr>np.percentile(eigval_fr,95)]
print('U-test spike:',mannwhitneyu(eig_big_en,eig_big_fr)[1])
print('U-test :',mannwhitneyu(eigval_en,eigval_fr)[1])




plt.hist(eigval_en,bins=100,alpha=0.5)
plt.hist(eigval_fr,bins=100,alpha=0.5)
plt.show()

w2vect_fr2=Word2vec('fastText/wiki.fr.vec',4000,random=0.5)
w2vect_wolof=Word2vec('fastText/wiki.wo.vec',4000,random=0.5)
w2vect_en=Word2vec('fastText/wiki.en.vec',4000,random=0.5)
sim_wolof=w2vect_wolof.vocab_similarity()
sim_wolof=(sim_wolof-np.min(sim_wolof))/(np.max(sim_wolof)-np.min(sim_wolof))



sim_fr2=w2vect_fr2.vocab_similarity()
sim_fr2=(sim_fr2-np.min(sim_fr2))/(np.max(sim_fr2)-np.min(sim_fr2))


sim_en=w2vect_en.vocab_similarity()
sim_en=(sim_en-np.min(sim_en))/(np.max(sim_en)-np.min(sim_en))



graph_fr=build_graph(sim_fr)
graph_fr2=build_graph(sim_fr2)
graph_en=build_graph(sim_en)
U_test_gpu_large(graph_fr,graph_fr2)

graph_wolof=build_graph(sim_wolof)

U_test_gpu(graph_fr,graph_fr)
U_test_gpu(graph_fr,graph_fr2)
U_test_gpu(graph_fr,graph_fr2)
U_test_gpu_large(graph_wolof,graph_fr)

deg_real1=np.sum(graph_1,0)
deg_real2=np.sum(graph_2,0)
from scipy.stats import mannwhitneyu
mannwhitneyu(deg_real1,deg_real2)


def langage_similarity(lang_A,lang_B,dict_size=4000,random=1):
    file_A='fastText/wiki.'+lang_A+'.vec'
    file_B='fastText/wiki.'+lang_B+'.vec'
    w2vect_A=Word2vec(file_A,dict_size,random)
    w2vect_B=Word2vec(file_B,dict_size,random)
    sim_A=w2vect_A.vocab_similarity()
    sim_A=(sim_A-np.min(sim_A))/(np.max(sim_A)-np.min(sim_A))
    sim_B=w2vect_B.vocab_similarity()
    sim_B=(sim_B-np.min(sim_B))/(np.max(sim_B)-np.min(sim_B))
    graph_A=build_graph(sim_A)
    graph_B=build_graph(sim_B)
    return U_test_gpu_large(graph_A,graph_B)

n_rep=20
sim_matrix=np.zeros((len(available_lang),len(available_lang)))
index2lang=dict(enumerate(available_lang))
lang2index = {v: k for k, v in index2lang.items()}
for lang_A in available_lang:
    for lang_B in available_lang:
        for rep in range(n_rep):
            i=lang2index[lang_A],lang2index[lang_B]
            j=lang2index[lang_A],lang2index[lang_B]
            sim_matrix[i,j]+=langage_similarity(lang_A,lang_B,dict_size=5000,random=0.5)[1]


for i in range(5):
    for j in range(5):
        if i==j:
             sim_matrix[i,j]*=1/5
        else:
            sim_matrix[i,j]*=1/10

#U_test_gpu_large(graph_1,graph_2)
count=0
for word in w2vect_fr.word2id:
    if word in w2vect_fr2.word2id:
        print(word)
        count+=1
print(count)


selected_word=np.random.randint(0,1000,10).tolist()
sim_en[selected_word,:][:,selected_word]


import matplotlib.pyplot as plt
plt.hist(deg_real1,alpha=0.5)
plt.hist(deg_real2,alpha=0.5)
plt.title("Distribution of degrees")
plt.show()





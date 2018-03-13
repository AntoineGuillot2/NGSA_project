#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:25:00 2018

@author: antoine
"""
import io
import numpy as np

class Word2vec():
    def __init__(self, fname, nmax=100000,random=1):
        self.load_wordvec(fname, nmax,random=random)
        self.wordvectors={}
        self.id2word = dict(enumerate(self.word2vec.keys()))
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.embeddings = np.concatenate(list(self.word2vec.values()),1)
    
    def load_wordvec(self, fname, nmax,random=1):
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                if word.isalnum() and np.random.uniform() <=random:
                    self.word2vec[word] = np.fromstring(vec, sep=' ').reshape((-1,1))
                if len(self.word2vec) == (nmax):
                    break

    def most_similar(self, w=None,embedding=None,K=5):
        if w is not None:
            vect=self.word2vec[w]
        elif embedding is not None:
            vect=embedding
        vect=vect/np.linalg.norm(vect)
        norm=np.sqrt(np.sum(self.embeddings**2,0))
        scores=np.dot((self.embeddings/norm).transpose(),vect)[:,0]
        closest=np.argsort(-scores)
        return [(self.id2word[i],scores[i]) for i in closest[:K]]
        

    def score(self, w1, w2):
        vect1=self.word2vec[w1]
        vect2=self.word2vec[w2]
        return np.dot(vect1.transpose(),vect2)[0][0]/(np.linalg.norm(vect1)*np.linalg.norm(vect2))
    
    def vocab_similarity(self):
        norm=np.sqrt(np.sum(self.embeddings**2,0))
        return np.dot((self.embeddings/norm).transpose(),self.embeddings/norm)
        
    def semantic_similarity(self):
        norm=np.sqrt(np.sum(self.embeddings**2,1)).reshape((-1,1))
        embeddings=self.embeddings/norm
        return np.dot((embeddings),embeddings.transpose())
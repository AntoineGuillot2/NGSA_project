#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:34:03 2018

@author: antoine
"""

from os import listdir
from os.path import isfile, join
available_lang = [f.split(".")[1] for f in listdir('fastText') if isfile(join('fastText', f))]

from tqdm import tqdm
from lib.langage_similarity import  GraphSpikedTST_PC, GraphSpikedTST_Cov


Utest,UtestSpike,idxA,idxB=GraphSpikedTST_PC('fastText/wiki.fr.vec','fastText/wiki.en.vec',10e4,plot_spectrum=True,random=1,n_graph=100)
Utest,UtestSpike,idxA,idxB=GraphSpikedTST_Cov('fastText/wiki.fr.vec','fastText/wiki.en.vec',10e4,plot_spectrum=True,random=1,n_graph=100)

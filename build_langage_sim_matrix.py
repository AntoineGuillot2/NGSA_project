#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:49:30 2018

@author: antoine
"""
from os import listdir
from os.path import isfile, join
available_lang = [f.split(".")[1] for f in listdir('fastText') if isfile(join('fastText', f))]

from tqdm import tqdm
from lib.langage_similarity import  U_test_langage_large,  U_test_langage,  U_test_langage_semantic
U_test_langage_semantic('fr','en',100000,plot_spectrum=True,random=0.1)

from skcuda import linalg as cuda_linalg


import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



sim_dict={}
n_rep=50
dict_size=256
random=1
for rep in tqdm(range(n_rep)):
    for lang_A in available_lang:
        for lang_B in available_lang:
            if (lang_A,lang_B) not in sim_dict:
                sim_dict[(lang_A,lang_B)]={'U_spike_linear':[],'U_spike':[],'U_test':[],'U_test_linear':[]}
            U_test,U_spike=U_test_langage(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike']+=[U_spike]
            sim_dict[(lang_A,lang_B)]['U_test']+=[U_test]
            
            U_test_linear,U_spike_linear=U_test_langage_large(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike_linear']+=[U_spike_linear]
            sim_dict[(lang_A,lang_B)]['U_test_linear']+=[U_test_linear]

save_object(sim_dict,'results/sim_dict_256')

sim_dict={}
n_rep=50
dict_size=1024
random=1
for rep in tqdm(range(n_rep)):
    for lang_A in available_lang:
        for lang_B in available_lang:
            if (lang_A,lang_B) not in sim_dict:
                sim_dict[(lang_A,lang_B)]={'U_spike_linear':[],'U_spike':[],'U_test':[],'U_test_linear':[]}
            U_test,U_spike=U_test_langage(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike']+=[U_spike]
            sim_dict[(lang_A,lang_B)]['U_test']+=[U_test]
            
            U_test_linear,U_spike_linear=U_test_langage_large(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike_linear']+=[U_spike_linear]
            sim_dict[(lang_A,lang_B)]['U_test_linear']+=[U_test_linear]

save_object(sim_dict,'results/sim_dict_1024')

sim_dict={}
n_rep=50
dict_size=2048
random=1
for rep in tqdm(range(n_rep)):
    for lang_A in available_lang:
        for lang_B in available_lang:
            if (lang_A,lang_B) not in sim_dict:
                sim_dict[(lang_A,lang_B)]={'U_spike_linear':[],'U_spike':[],'U_test':[],'U_test_linear':[]}
            U_test,U_spike=U_test_langage(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike']+=[U_spike]
            sim_dict[(lang_A,lang_B)]['U_test']+=[U_test]
            
            U_test_linear,U_spike_linear=U_test_langage_large(lang_A,lang_B,dict_size,random)
            sim_dict[(lang_A,lang_B)]['U_spike_linear']+=[U_spike_linear]
            sim_dict[(lang_A,lang_B)]['U_test_linear']+=[U_test_linear]

save_object(sim_dict,'results/sim_dict_2048')

print(U_test_langage('fr','sq',2000))
print(U_test_langage_large('fr','sq',2000))
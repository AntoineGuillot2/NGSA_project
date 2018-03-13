#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:52:23 2018

@author: antoine
"""

import numpy as np
def build_graph(sim_matrix):
    random_mat=np.random.uniform(size=sim_matrix.shape)
    random_mat=np.tril(random_mat)+np.transpose(np.tril(random_mat))
    return (1-sim_matrix <= random_mat).astype(float)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:12:16 2025

@author: Louis Brezin

This code runs a simulation of Nonlinear diffusion and writes to a result file
"""

import NLDv3
import numpy as np
import time

accuracy = 0.35 #High accuracy for precise profiles
param={}
start = time.time()
Db = np.logspace(-1, 1,num=21)
for al in [0]:
    for values in [0.01]:
        param = {
            'L' : 3000,
            'BC' : 'Dir',
            't0' : 0.0,
            'Dn' : 1.0,
            'Db' : values,
            'gamma' : 1,
            'alpha' : al,
            'method' : 'linear',
            'K' : 5*10**(-2),
            'nm' : 5*10**(-2),
            'beta' :1,
            'dt' : 0.0,
            'accuracyX': accuracy,
            'accuracyT': accuracy,
            'n0' : 1
        }
        #Run and save the data
        if values <= 1:
            step = 100
        else:
            step = 100
        NLDv3.run_and_save(param, saveFile = 'revisions.h5', csvFile = 'lookup_revisions.csv', save_every = 10, step=step, threshold_fraction = 0.05)
        
stop = time.time()
print(f"Total time : {stop-start}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:09:05 2019

@author: David Chong Tian Wei
"""

import pandas as pd
import numpy as np

def pre_process_ventilation_data(path, cols):
    '''
    Loads ventilation data and fills in NA values via linear interpolation
    Input: (String) path to file, (List) Column names or integers for Pressure, and Flow respectively in that order
    Output: (pd.DataFrame) dataframe of the loaded data
    '''
    data = pd.read_csv(path, usecols=cols)
    # Impute missing values for pressure, flow, volume
    for i in range(0,data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i].interpolate()
    return data

def correct_baseline(data, window):
    for i in range(0,int(len(data)/window)):
        if (i+1)*window < data.shape[0]:
            data[(i*window):((i+1)*window)] = data[(i*window):((i+1)*window)] - np.mean(data[(i*window):((i+1)*window)])
        else:
            data[(i*window):] = data[(i*window):] - np.mean(data[(i*window):])
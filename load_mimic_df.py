#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:10:22 2020

@author: scro3517
"""

import numpy as np
import pandas as pd
import random

def modify_df():
    """ Load Dataframe as an Iterable """
    iter_csv = pd.read_csv('/home/scro3517/Desktop/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv', iterator=True, chunksize=5000)
    """ Filter DF in an Online Manner Keeping Certain Rows """
    df = pd.concat([chunk[(chunk['CATEGORY'] == 'ECG').astype(bool) & ~chunk.TEXT.str.contains('ECG interpreted by ordering physician').astype(bool)] for chunk in iter_csv])
    """ Determine How Many of the Reports Contain the Following Keywords """
    condition = 'Sinus rhythm|sinus rhythm|Sinus bradycardia|sinus bradycardia|bradycardia|Sinus tachycardia|sinus tachycardia|tachycardia|Atrial fibrillation|atrial fibrillation|Atrial flutter|atrial flutter'
    """ Bucketize Text According to Groups """
    mapping_dict = {'Sinus rhythm|sinus rhythm': 0, 'Sinus bradycardia|sinus bradycardia|bradycardia': 1,
                    'Sinus tachycardia|sinus tachycardia|tachycardia': 2,
                    'Atrial fibrillation|atrial fibrillation': 3,
                    'Atrial flutter|atrial flutter': 4}
    
    """ (1) Assign Each Text Report to a Particular Category """
    df_subset = df[df.TEXT.str.contains(condition)].reset_index()
    group_df = pd.Series(np.zeros((df_subset.shape[0])))
    for key,value in mapping_dict.items():
        bool_series = df_subset.TEXT.str.contains(key)
        group_df[bool_series] = value
    """ Add Category Column to DataFrame """
    df_subset['TextCategory'] = group_df
    
    """ (2) Assign Reports to Training and Validation Sets  """
    random.seed(0) #seed is important since we load this multiple times
    shuffled_indices = random.sample(list(range(df_subset.shape[0])),df_subset.shape[0])
    training_length = int(len(shuffled_indices)*0.8)
    #training_indices = shuffled_indices[:training_length]
    validation_indices = shuffled_indices[training_length:]
    df_subset['Phase'] = 'train'
    #df_subset['Phase'][df_subset.index.isin(training_indices)] = 'train'
    df_subset['Phase'][df_subset.index.isin(validation_indices)] = 'val'    
    
    return df_subset
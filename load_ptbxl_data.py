#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:22:14 2020

@author: scro3517
"""

import numpy as np
import os
import pickle
import wfdb
from tqdm import tqdm
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder
import ast
from operator import itemgetter
from itertools import compress

#%%
def load_paths(basepath):
    """ File Names to Load """
    folders1 = os.listdir(basepath)
    pth_to_folders = [folder for folder in folders1 if 'records%i' % fs in folder] #records100 contains 100Hz data
    pth_to_folders = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder))]
    files = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder))]
    paths_to_files = [os.path.join(basepath,file.split('.hea')[0]) for file in files if '.hea' in file]
    return paths_to_files
#%%
def modify_df(basepath,code_of_interest='diagnostic_class',output_type='single'):
    """ Database with Patient-Specific Info """
    df = pd.read_csv(os.path.join(basepath,'ptbxl_database.csv'),index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    """ Database with Label Information """
    codes_df = pd.read_csv(os.path.join(basepath,'scp_statements.csv'),index_col=0)
        
    if output_type == 'single':
        encoder = LabelEncoder()
    elif output_type == 'multi':
        encoder = MultiLabelBinarizer()
    
    def aggregate_diagnostic(y_dic):
        """ Map Label To Diffeent Categories """
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))
    
    aggregation_df = codes_df
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
    
    """ Obtain Superdiagnostic Label(s) """
    df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
    """ Obtain Number of Superdiagnostic Label(s) Per Recording """
    df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    
    """ Return Histogram of Each Label """
    min_samples = 0
    counts = pd.Series(np.concatenate(df.superdiagnostic.values)).value_counts()
    counts = counts[counts > min_samples]
    
    """ Obtain Encoded Label as New Column """
    if output_type == 'single':
        df = df[df.superdiagnostic_len == 1] # ==1 OR > 0
        df.superdiagnostic = df.superdiagnostic.apply(lambda entry:entry[0])
        encoder.fit(df.superdiagnostic.values)
        df['superdiagnostic_label'] = encoder.transform(df.superdiagnostic.values)
    elif output_type == 'multi':
        df = df[df.superdiagnostic_len > 0]
        encoder.fit(list(map(lambda entry: [entry],counts.index.values)))
        multi_hot_encoding_df = pd.DataFrame(encoder.transform(df.superdiagnostic.values),index=df.index,columns=encoder.classes_.tolist()) 
        df = pd.merge(df,multi_hot_encoding_df,on=df.index).drop(['key_0'],axis=1)
    
    return df

#%%
def obtain_phase_to_paths_dict(df,paths_to_files,input_type='single'):
    train_fold = np.arange(0,8)
    val_fold = [9]
    test_fold = [10]
    phases = ['train','val','test']
    folds = [train_fold,val_fold,test_fold]
    
    """ Obtain Patient IDs """
    phase_to_pids = dict()
    for phase,fold in zip(phases,folds):
        current_ecgid = df[df.strat_fold.isin(fold)].index.tolist() #index is ecg_id by default when loading csv
        current_ecgid = list(map(lambda entry:int(entry),current_ecgid))
        phase_to_pids[phase] = current_ecgid
    
    paths_to_ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0]),paths_to_files))
    paths_to_ids_df = pd.Series(paths_to_ids)
    """ Obtain Paths For Each Phase """
    phase_to_paths = dict()
    phase_to_leads = dict()
    phase_to_segment_number = dict()
    nsegments = 2
    
    """ Determine Whether Each Input Contains Single Or Multiple Leads """
    if input_type == 'single': #input will contain a single lead i.e. X = (B,1,L)
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        nleads = 12
    elif input_type == 'multi': #input will contain all leads i.e. X = (B,12,L)
        leads = ['All'] #filler
        nleads = 1
        
    for phase,pids in phase_to_pids.items():
        """ Obtain Paths For All Leads """
        paths = list(compress(paths_to_files,paths_to_ids_df.isin(pids).tolist()))
        paths_for_all_leads = np.repeat(paths,nleads*nsegments)
        """ Obtain Leads Label """
        leads = np.repeat(leads,nsegments)
        all_leads = np.tile(leads,len(paths))
        """ Obtain Segment Numbers (First Segment and Second Segment of Recording) """
        segment_number = [0,1]
        segment_number = np.tile(segment_number,nleads)
        all_segment_numbers = np.tile(segment_number,len(paths))
        """ Assign Paths and Leads Labels """
        phase_to_paths[phase] = paths_for_all_leads
        phase_to_leads[phase] = all_leads
        phase_to_segment_number[phase] = all_segment_numbers
    
    return phase_to_paths, phase_to_leads, phase_to_segment_number

#%%
basepath = '/mnt/SecondaryHDD/PTB-XL'
fs = 500 #options: 500 | 100
""" Identify Codes of Interest """
code_of_interest = 'diagnostic_class' # options: 'rhythm' | 'diagnostic_class' | 'all' 
""" Identify Encoder Based on Output Type """
input_type = 'multi' #options: 'single' is single lead input | 'multi' is multiple lead input 
output_type = 'single' #options: 'single' is single output | 'multi' is multi-output

if __name__ == '__main__':
    paths_to_files = load_paths(basepath)
    df = modify_df(basepath,code_of_interest=code_of_interest,output_type=output_type)
    phase_to_paths, phase_to_leads, phase_to_segment_number = obtain_phase_to_paths_dict(df,paths_to_files,input_type=input_type)
    
    savepath = os.path.join(basepath,'patient_data','%s_input' % input_type,'%s_output' % output_type)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    """ Save Paths """
    with open(os.path.join(savepath,'phase_to_paths.pkl'),'wb') as f:
        pickle.dump(phase_to_paths,f)
    """ Save Leads """
    with open(os.path.join(savepath,'phase_to_leads.pkl'),'wb') as f:
        pickle.dump(phase_to_leads,f)
    """ Save Segment Number """
    with open(os.path.join(savepath,'phase_to_segment_number.pkl'),'wb') as f:
        pickle.dump(phase_to_segment_number,f)










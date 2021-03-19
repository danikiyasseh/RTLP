#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:53:51 2020

@author: scro3517
"""

import numpy as np
import os
import pickle
from tqdm import tqdm
import pandas as pd
from itertools import compress

#%%
def load_paths(basepath):
    """ File Names to Load """
    files = os.listdir(os.path.join(basepath,'ECGDataDenoised'))
    paths_to_files = [os.path.join(basepath,'ECGDataDenoised',file) for file in files]
    return paths_to_files
#%%
def modify_df(output_type='single'):
    basepath = '/mnt/SecondaryHDD/chapman_ecg'
    """ Database with Patient-Specific Info """
    df = pd.read_csv(os.path.join(basepath,'Diagnostics.csv'))
    dates = df['FileName'].str.split('_',expand=True).iloc[:,1]
    dates.name = 'Dates'
    dates = pd.to_datetime(dates)
    df = pd.concat((df,dates),1)
    
    """ Combine Rhythm Labels """
    old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
    new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
    df['Rhythm'] = df['Rhythm'].replace(old_rhythms,new_rhythms)

    """ Add Date Column """
    def combine_dates(date):
        new_dates = ['All Terms']#use this for continual learning dataset ['Term 1','Term 2','Term 3']
        cutoff_dates = ['2019-01-01']##use this for continual learning dataset ['2018-01-16','2018-02-09','2018-12-30']
        cutoff_dates = [pd.Timestamp(date) for date in cutoff_dates]
        for t,cutoff_date in enumerate(cutoff_dates):
            if date < cutoff_date:
                new_date = new_dates[t]
                break
        return new_date
    
    df['Dates'] = df['Dates'].apply(combine_dates)

    """ Replace Certain Column Names With Questions """
    question_prefix = 'What is the '
    columns = df.loc[:,'VentricularRate':'TOffset'].columns
    new_columns = list(map(lambda column: question_prefix + column + '?',columns))
    column_mapping = dict(zip(columns,new_columns))
    df.rename(columns=column_mapping,inplace=True)
    
    """ Scale Values in the Columns """
    min_values = df[new_columns].min()
    max_values = df[new_columns].max()
    df[new_columns] = 2*(df[new_columns] - min_values)/(max_values - min_values) - 1

    return df, max_values, min_values

#%%
def obtain_phase_to_paths_dict(df,paths_to_files,nquestions=1,input_type='single'):
    phases = ['train','val','test']
    phase_fractions = [0.6, 0.2, 0.2]
    phase_fractions_dict = dict(zip(phases,phase_fractions))
    terms = ['All Terms']##use this for continual learning dataset ['Term 1','Term 2','Term 3']
    
    """ Obtain Mapping from Phase to List of Filenames (From DataFrame) """
    phase_to_term_to_filenames = dict()
    for term in terms:
        phase_to_term_to_filenames[term] = dict()
        term_patients = df['FileName'][df['Dates'] == term]
        random_term_patients = term_patients.sample(frac=1,random_state=0)
        start = 0
        for phase,fraction in phase_fractions_dict.items():
            if phase == 'test':
                phase_patients = random_term_patients.iloc[start:].tolist() #to avoid missing last patient due to rounding
            else:
                npatients = int(fraction*len(term_patients))
                phase_patients = random_term_patients.iloc[start:start+npatients].tolist()
                        
            phase_to_term_to_filenames[term][phase] = phase_patients
            start += npatients
    
    """ Obtain Filenames Only (Without Entire Path) """
    paths_to_filenames = list(map(lambda path: path.split('/')[-1].split('.csv')[0], paths_to_files))
    paths_to_filenames_df = pd.Series(paths_to_filenames)
    """ Obtain Paths For Each Phase """
    phase_to_paths = dict()
    phase_to_leads = dict()
    phase_to_segment_number = dict()
    phase_to_questions = dict()
    nsegments = 2
    
    """ Determine Whether Each Input Contains Single Or Multiple Leads """
    if input_type == 'single': #input will contain a single lead i.e. X = (B,1,L)
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        nleads = 12
    elif input_type == 'multi': #input will contain all leads i.e. X = (B,12,L)
        leads = ['All'] #filler
        nleads = 1 #to avoid repeating path an unnecessary number of times
        
    for term,phase_to_filenames in phase_to_term_to_filenames.items():    
        for phase,filenames in tqdm(phase_to_filenames.items()):
            """ Obtain Paths For All Leads """ #(Are you in dataframe?)
            paths = list(compress(paths_to_files,paths_to_filenames_df.isin(filenames).tolist()))
            """ Check if Path Data is Valid and Return Only Valid Paths """
            print(len(paths))
            paths = check_if_data_is_valid(paths)
            print(len(paths))
            paths_for_all_leads = np.repeat(paths,nleads*nsegments*nquestions)
            """ Obtain Leads Label """
            repeated_leads = np.repeat(leads,nsegments*nquestions) #for each path
            all_leads = np.tile(repeated_leads,len(paths))
            """ Obtain Segment Numbers (First Segment and Second Segment of Recording) """
            segment_number = [0,1]
            segment_number = np.tile(segment_number,nleads*nquestions) #for each path
            all_segment_numbers = np.tile(segment_number,len(paths))
            """ Obtain Questions """
            questions = list(compress(df.columns.tolist(),['What' in column for column in df.columns]))
            questions = np.tile(questions,nleads*nsegments) #for each path
            all_questions = np.tile(questions,len(paths))            
            """ Assign Paths and Leads Labels """
            phase_to_paths[phase] = paths_for_all_leads
            phase_to_leads[phase] = all_leads
            phase_to_segment_number[phase] = all_segment_numbers
            phase_to_questions[phase] = all_questions
    
    return phase_to_paths, phase_to_leads, phase_to_segment_number, phase_to_questions

def check_if_data_is_valid(paths):
    new_paths = []
    for path in tqdm(paths):
        data = pd.read_csv(path)
        data_np = data.to_numpy()
        """ Is Matrix Entirely Zeros? """
        all_zero_condition = np.sum(data_np == 0) == data.size
        """ Is There a NaN in the Data? """
        nan_condition = np.sum(np.isnan(data_np)) > 0
        if all_zero_condition or nan_condition:
            continue
        else:
            new_paths.append(path)
    return new_paths
#%%
basepath = '/mnt/SecondaryHDD/chapman_ecg'

""" Identify Encoder Based on Output Type """
input_type = 'multi' #options: 'single' is single lead input | 'multi' is multiple lead input 
output_type = 'single' #options: 'single' is single output | 'multi' is multi-output
goal = 'VQA' #options: 'VQA' | 'Supervised'

if __name__ == '__main__':
    paths_to_files = load_paths(basepath)
    df, max_values, min_values = modify_df(output_type=output_type)
    if goal == 'VQA': #generate data for VQA setting
        nquestions = sum(['What' in column for column in df.columns])
    else: #generate data for traditional supervised setting
        nquestions = 1
    phase_to_paths, phase_to_leads, phase_to_segment_number, phase_to_questions = obtain_phase_to_paths_dict(df,paths_to_files,nquestions=nquestions,input_type=input_type)
    
    savepath = os.path.join(basepath,'patient_data',goal,'%s_input' % input_type,'%s_output' % output_type)
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
    """ Save Questions """
    with open(os.path.join(savepath,'phase_to_questions.pkl'),'wb') as f:
        pickle.dump(phase_to_questions,f)
#








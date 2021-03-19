#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:05:50 2020

@author: scro3517
"""

#import spacy
import os
import numpy as np
from load_ptbxl_data import modify_df
from tqdm import tqdm
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

from googletrans import Translator
import httpx
timeout = httpx.Timeout(None) # 5 seconds timeout
translator = Translator(timeout=timeout)

from langdetect import detect
#%%

def load_ptbxl_df():
    """ Load Database With Text """
    df = modify_df('/mnt/SecondaryHDD/PTB-XL',output_type='single') #output_type options: 'single' | 'multi'
    return df

def make_changes_to_string(text):
    if 'sinusrhythmus' in text:
        text = text.replace('sinusrhythmus','sinusrhythm')
    
    if 'normales' in text:
        text = text.replace('normales','normal')
    
    return text
    
def hard_code_text_changes_df(df):
    df.report = df.report.apply(make_changes_to_string)
    return df

#%%
def obtain_translations_df(df,dest_lang):
    """ Translate Text Into Desired Language """
    
    """ Set Up Empty DF to Populate """
    translations_df = pd.DataFrame(np.zeros((df.shape[0],1)),index=df.index)
    """ Always Translating From Original Report """
    r = 0
    for report in tqdm(df.report): 
        """ Check if Report is Empty """
        if len(report.split()) != 0:
            #""" Detect Source Language """
            #src_lang = detect(report)
            """ Translate From Source to Target Language """
            t = translator.translate(report, src='auto', dest=dest_lang)
            text = t.text
        else:
            text = ''
        """ Populate Translations DF with Translation """
        translations_df.iloc[r] = text
        r += 1
    return translations_df

def expand_df(df,translations_df,dest_lang):
    """ Append Translations As New Column in DF """
    new_column_name = '%s_report' % dest_lang
    df[new_column_name] = translations_df
    return df

#%%
parser.add_argument('-lang',nargs='+',)    
args = parser.parse_args()
translation_round = 2 #1 means translate based on raw report #2 means translate in an iterative manner to close the gaps

if __name__ == '__main__':
    """ Round 1 """
    if translation_round == 1:
        
        """ First Time Loading - i.e., if no translations have been performed yet """
        df = load_ptbxl_df()
        """ Hard Code Some Changes to Original Report """
        df = hard_code_text_changes_df(df)
        """ Second Time Loading - i.e., if n > 0 translationns have been performed already """
        #df = pd.read_csv('/mnt/SecondaryHDD/PTB-XL/multi_lingual_df.csv')
        """ Identify Destination Languages """
        #dest_lang_list = ['it','pt','es','el','ja','zh-CN','en','de','fr']
        dest_lang_list = args.lang
        for dest_lang in dest_lang_list:
            print('\nTranslating to %s...' % dest_lang)
            translations_df = obtain_translations_df(df,dest_lang)
            df = expand_df(df,translations_df,dest_lang)
            """ Save The Translations DataFrame (Frequently To Avoid Losing Progress) """
            translations_df.to_csv(os.path.join('/mnt/SecondaryHDD/PTB-XL','%s_df.csv' % dest_lang))
            #""" Save The Multi-lingual DataFrame (Frequently To Avoid Losing Progress) """
            #df.to_csv(os.path.join('/mnt/SecondaryHDD/PTB-XL','multi_lingual_df.csv'))
    elif translation_round == 2:
        """ Round 2 """
        
    """ Round 2 of Translations to Catch Those Not Translated The First Time Round """
    #dest_lang_list = ['de', 'en', 'fr', 'pt', 'it', 'es', 'el', 'ja', 'zh-CN']
    dest_lang_list = args.lang
    for dest_lang in dest_lang_list:
        """ Load Translated Report """
        translations_df = pd.read_csv('/mnt/SecondaryHDD/PTB-XL/%s_df_round3.csv' % dest_lang,index_col='ecg_id')
        translations_df.columns = ['report']
        
        """ Identify Reports Where Detected Language != Desired Language """
        #src_langs = []
        indices_to_translate = []
        r = 0
        for text in tqdm(translations_df.report):
            if isinstance(text,str):
                src_lang = detect(text) #cannot identify correct language accurately, but can be used to identify whether sentence is entirely in desired language 
                if src_lang != dest_lang.lower():
                    indices_to_translate.append(r)
                    #src_langs.append(src_lang)
            r += 1
        
        """ Translate Only Those That Need Translating """
        for index in tqdm(indices_to_translate):
            text = translations_df.report.iloc[index]
            translation = translator.translate(text, src='de', dest=dest_lang).text
            translations_df.iloc[index] = translation
        
        """ Save Translations DF """
        translations_df.to_csv(os.path.join('/mnt/SecondaryHDD/PTB-XL','%s_df_round4.csv' % dest_lang))
        


    
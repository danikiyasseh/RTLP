#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:57:52 2020

@author: scro3517
"""

import pickle
import spacy
import numpy as np
from load_ptbxl_data import modify_df
import pandas as pd
import load_chapman_ecg_v2
import load_mimic_df
from itertools import compress
from translate_brazil_text import load_brazil_df

from operator import itemgetter
import random
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

#%%
""" Must Download These in Terminal """
#python -m spacy download zh_core_web_md
#python -m spacy download fr_core_news_md
#python -m spacy download el_core_news_md
#python -m spacy download it_core_news_md
#python -m spacy download ja_core_news_md
#python -m spacy download pt_core_news_md
#python -m spacy download es_core_news_md

#%%
""" Needed When Initializing The Word Embeddings (prepare_model.py) AND When Loading Data (prepare_dataset.py) """

def load_ptbxl_df(dest_lang_list):
    """ Load Database With Text """
    df = modify_df('/mnt/SecondaryHDD/PTB-XL',output_type='single') #output_type options: 'single' | 'multi'
    #dest_lang_list = ['de', 'en', 'fr', 'pt', 'it', 'es', 'el', 'ja', 'zh-CN']
    for dest_lang in dest_lang_list:
        translations_df = pd.read_csv('/mnt/SecondaryHDD/PTB-XL/%s_df_round4.csv' % dest_lang,index_col='ecg_id')
        column_of_interest = '%s_report' % dest_lang
        df[column_of_interest] = translations_df
    #df = pd.read_csv('/mnt/SecondaryHDD/PTB-XL/multi_lingual_df_old.csv',index_col='ecg_id')
    return df

#def load_brazil_df():
#    """ Load Database With Text """
#    df = pd.read_csv('/mnt/SecondaryHDD/Brazil_ECG/ECG_REPORT/ecg_reports_anonymized_round2.csv') #more anonymized version than round1
#    """ Load Paths From Continual Setting for Consistency's Sake """
#    with open('/mnt/SecondaryHDD/Brazil_ECG/patient_data/year_to_paths.pkl','rb') as f:
#        year_to_paths = pickle.load(f)
#    
#    """ Introduce Phase Column to Know Which Tokens to Use for Token Mapping """
#    train_ids = []
#    for year,paths in year_to_paths.items():
#        current_year_paths = paths['train']
#        current_year_ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0].split('TNMG')[1]),current_year_paths))
#        train_ids.extend(current_year_ids)
#    
#    df['Phase'] = df['ID_EXAME'].isin(train_ids).astype(int)
#    df['Phase'].replace({1:'train',0:'not train'},inplace=True)
#    
#    return df

class modify_text(object):

    def __init__(self,dest_lang,dataset_name):
        super(modify_text,self).__init__()
        self.dest_lang = dest_lang
        self.dataset_name = dataset_name
    
    def add_start_end_symbols(self,df):
        """ This Needs to Be Applied to Matrix Before Training """
        column_of_interest = '%s_report' % self.dest_lang

        if self.dataset_name in ['brazil']:
            df['new_report'] = df[column_of_interest].apply(self.add_start_end_symbols_to_single_report)
            df['new_report'] = df.new_report.apply(self.space_out_text_in_single_report)
        else:
            df['new_report'] = df[column_of_interest].apply(self.add_start_end_symbols_to_single_report)

        return df
    
    def obtain_start_and_end_dict(self):
        start_dict = {'de': 'START',
                        'en': '/START',
                        'fr': 'DÉBUT',
                        'pt': 'COMEÇAR',
                        'it': 'INIZIO',
                        'es': 'COMIENZO',
                        'el': 'ΑΡΧΗ',
                        'ja': '開始',
                        'zh-CN': '开始'}
    
        end_dict = {'de': 'ENDE',
                        'en': '/END',
                        'fr': 'FIN',
                        'pt': 'FIM',
                        'it': 'FINE',
                        'es': 'FIN',
                        'el': 'ΤΕΛΟΣ',
                        'ja': '終わり',
                        'zh-CN': '结束'}
        return start_dict, end_dict
    
    """ Add Start and End Symbol to Each Report Text """
    def add_start_end_symbols_to_single_report(self,text):
        start_dict, end_dict = self.obtain_start_and_end_dict()
        start = start_dict[self.dest_lang]
        end = end_dict[self.dest_lang]
        new_text = start + ' ' + str(text) + ' ' + end
        return new_text
    
    """ Space Out Text Between Punctuations and Words To Obtain Better Tokens Later """
    def space_out_text_in_single_report(self,text):
        new_text = ''
        for t in text:
            if t in [':','.']:
                t = ' ' + t + ' '
            new_text += t
        return new_text

def load_nlp(dest_lang):
    mapping_dict = {'de': 'de_core_news_md',
                    'en': 'en_core_web_md',
                    'fr': 'fr_core_news_md',
                    'pt': 'pt_core_news_md',
                    'it': 'it_core_news_md',
                    'es': 'es_core_news_md',
                    'el': 'el_core_news_md',
                    'ja': 'ja_core_news_md',
                    'zh-CN': 'zh_core_web_md'}
    
    nlp = spacy.load(mapping_dict[dest_lang])
    nlp.max_length = 10000000
    return nlp

#%%
""" Needed When Initializing The Word Embeddings (prepare_model.py) """

def shrink_data(df_subset,fraction=1):
    tot_rows = df_subset.shape[0]
    nrows = int(tot_rows*fraction)
    random.seed(0) #to make sure we always obtain SAME shrunken dataset/vocabulary for reproducibility 
    indices = random.sample(list(np.arange(tot_rows)),nrows)
    df_subset = df_subset.iloc[indices,:]
    return df_subset

def obtain_ptbxl_language_tokens(df,dest_lang,fraction=1):
    """ Obtain Language Specific Column and Add Special Tokens """
    text_modifier = modify_text(dest_lang,'ptbxl')
    df = text_modifier.add_start_end_symbols(df)
    """ Load the NLP For Tokenization """
    nlp = load_nlp(dest_lang)
    """ Obtain Training Subset (Embeddings Guided by Training Set Only) """
    df_subset = df[df.strat_fold<=8]
    """ Shrink Dataset if Applicable """
    df_subset = shrink_data(df_subset,fraction)
    """ Join All Report Text As One Large String """
    report_text = ' '.join(df_subset.new_report.values)
    """ Split According to Whitespace """ 
    report_text = report_text.split() #step 1 for each instance during training/inference
    """ Remove Duplicate Text """
    report_text = list(np.unique(report_text))
    """ Convert List to String """
    report_text = ' '.join(report_text) #step 2 for each instance during training/inference
    print('Obtaining Tokens...')
    """ Apply Tokenizer to Text """
    doc = nlp(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
    """ Remove Punctuation and Return Lemmatized Word """
    tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
    """ Remove Duplicates Again """
    tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference
    return tokens

def obtain_brazil_language_tokens(df,dest_lang,fraction=1):
    """ Obtain Language Specific Column and Add Special Tokens """
    text_modifier = modify_text(dest_lang,'brazil')
    df = text_modifier.add_start_end_symbols(df)
    nlp = load_nlp(dest_lang)
    """ Obtain Training Subset (Embeddings Guided by Training Set Only) """
    df_subset = df[df.Phase == 'train']
    """ Shrink Dataset if Applicable """
    df_subset = shrink_data(df_subset,fraction)
    """ Join All Report Text As One Large String """
    report_text = ' '.join(df_subset.new_report.values)
    """ Split According to Whitespace """ 
    report_text = report_text.split() #step 1 for each instance during training/inference
    """ Remove Duplicate Text """
    report_text = list(np.unique(report_text))
    """ Convert List to String """
    report_text = ' '.join(report_text) #step 2 for each instance during training/inference
    print('Obtaining Tokens...')        
    """ Apply Tokenizer to Text """
    doc = nlp(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
    """ Remove Punctuation and Return Lemmatized Word """
    #tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
    tokens = [token.text.lower() for token in doc if token.is_alpha == True or token.text in ['/START','/END']] #step 4 for each instance during training/inference
    """ Remove Duplicates Again """
    tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference
    return tokens

def obtain_token_language_mapping(tokens):
    """ Create Mapping from Tokens to ID/Index """
    token2id_dict = dict(zip(tokens,np.arange(1,len(tokens)+1))) #starting at 1 b/c 0 is reserved for padded regions not to be considered in loss (later)
    """ Include OOD Mapping Dict (In Case Test Set Has Unseen Token) """
    ood2id_dict = {'OOD':len(tokens)+1}
    """ Include PAD Mapping Dict """
    pad2id_dict = {'PAD':0}
    """ Include MASK Mapping Dict """
    mask2id_dict = {'MASK':-1}
    """ Final Mapping Dict From Tokens to ID/Index """
    token2id_dict = {**pad2id_dict,**token2id_dict,**ood2id_dict,**mask2id_dict}
    return token2id_dict
    
def obtain_token_mapping(dataset_name,dest_lang_list,fraction=1):
    """ Returns Mapping Between Tokens and ID 
    Args:
        df (Pandas): database with report text
    
    Returns:
        token2id_dict (dict): 
    """
    
    complete_token2id_dict = dict()
    if 'ptbxl' in dataset_name:
        """ Load DF """
        print('Loading Dataframe...')
        df = load_ptbxl_df(dest_lang_list)
        for dest_lang in dest_lang_list:
            print(dest_lang)
            tokens = obtain_ptbxl_language_tokens(df,dest_lang,fraction)
            token2id_dict = obtain_token_language_mapping(tokens)
            complete_token2id_dict[dest_lang] = token2id_dict
            
#        """ Obtain Language Specific Column and Add Special Tokens """
#        text_modifier = modify_text(dest_lang)
#        print('Adding Start/End Symbols...')
#        df = text_modifier.add_start_end_symbols(df)
#        """ Load the NLP For Tokenization """
#        nlp = load_nlp(dest_lang)
#        """ Obtain Training Subset (Embeddings Guided by Training Set Only) """
#        df_subset = df[df.strat_fold<=8]
#        """ Join All Report Text As One Large String """
#        report_text = ' '.join(df_subset.new_report.values)
#        """ Split According to Whitespace """ 
#        report_text = report_text.split() #step 1 for each instance during training/inference
#        """ Remove Duplicate Text """
#        report_text = list(np.unique(report_text))
#        """ Convert List to String """
#        report_text = ' '.join(report_text) #step 2 for each instance during training/inference
#        print('Obtaining Tokens...')
#        """ Apply Tokenizer to Text """
#        doc = nlp(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
#        """ Remove Punctuation and Return Lemmatized Word """
#        tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
#        """ Remove Duplicates Again """
#        tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference
        #nlp = nlp_de
    elif 'chapman' in dataset_name:
        df, max_values, min_values = load_chapman_ecg_v2.modify_df()
        nlp_en = spacy.load('en_core_web_sm')
        report_text = list(compress(df.columns.tolist(),['What' in column for column in df.columns]))
        """ Convert List to String """
        report_text = ' '.join(report_text) #step 2 for each instance during training/inference
        """ Apply Tokenizer to Text """
        doc = nlp_en(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
        """ Remove Punctuation and Return Lemmatized Word """
        tokens = [token.text for token in doc]# if token.is_punct == False] #step 4 for each instance during training/inference
        """ Remove Duplicates Again """
        tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference 
        #nlp = nlp_en
    elif 'mimic' in dataset_name:
        print('Loading Dataframe...')
        df = load_mimic_df.modify_df()
        df_subset = df[df.Phase == 'train']
        """ Apply Tokenizer to Text """
        print('Applying Spacy Tokenizer...')
        docs = [tokenizer(' '.join(text.split())) for text in df_subset.TEXT]
        """ Remove Punctuation and Return Lemmatized Word """
        tokens = [token.lower_ for doc in docs for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
        """ Remove Duplicates Again """
        tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference
        """ Remove Anonymized Patient/Physician Names """
        tokens = [token for token in tokens if ('[' not in token) & (']' not in token)]
    elif 'brazil' in dataset_name:
        print('Loading Dataframe...')
        df = load_brazil_df(dest_lang_list)
        for dest_lang in dest_lang_list:
            print(dest_lang)
            tokens = obtain_brazil_language_tokens(df,dest_lang,fraction)
            token2id_dict = obtain_token_language_mapping(tokens)
            complete_token2id_dict[dest_lang] = token2id_dict
#            """ Obtain Language Specific Column and Add Special Tokens """
#            text_modifier = modify_text(dest_lang)
#            print('Adding Start/End Symbols...')
#            df = text_modifier.add_start_end_symbols(df)
#            nlp = load_nlp(dest_lang)
#            """ Obtain Training Subset (Embeddings Guided by Training Set Only) """
#            df_subset = df[df.Phase == 'train']
#            """ Join All Report Text As One Large String """
#            report_text = ' '.join(df_subset.new_report.values)
#            """ Split According to Whitespace """ 
#            report_text = report_text.split() #step 1 for each instance during training/inference
#            """ Remove Duplicate Text """
#            report_text = list(np.unique(report_text))
#            """ Convert List to String """
#            report_text = ' '.join(report_text) #step 2 for each instance during training/inference
#            print('Obtaining Tokens...')        
#            """ Apply Tokenizer to Text """
#            doc = nlp(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
#            """ Remove Punctuation and Return Lemmatized Word """
#            tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
#            """ Remove Duplicates Again """
#            tokens = list(map(lambda x: str(x),np.unique(tokens))) #step 5 for each instance during training/inference
            
#        """ Create Mapping from Tokens to ID/Index """
#        token2id_dict = dict(zip(tokens,np.arange(1,len(tokens)+1))) #starting at 1 b/c 0 is reserved for padded regions not to be considered in loss (later)
#        """ Include OOD Mapping Dict (In Case Test Set Has Unseen Token) """
#        ood2id_dict = {'OOD':len(tokens)+1}
#        """ Include PAD Mapping Dict """
#        pad2id_dict = {'PAD':0}
#        """ Include MASK Mapping Dict """
#        mask2id_dict = {'MASK':-1}
#        """ Final Mapping Dict From Tokens to ID/Index """
#        token2id_dict = {**pad2id_dict,**token2id_dict,**ood2id_dict,**mask2id_dict}
        
#        """ Dict For All Languages """
#        complete_token2id_dict[dest_lang] = token2id_dict
    
    return complete_token2id_dict, df

""" Remember, for OOD, you need to compare tokens to vocab to see if they are in it, i.e., [token if token in vocab else 'OOD' for token in tokens] """
#%%
#""" Create Vocab From These Tokens """
#from spacy.vocab import Vocab
#vocab = Vocab(strings=tokens)

#from spacy.tokenizer import Tokenizer
#from spacy.lang.de import German
#nlp_de = German()
#tokenizer = Tokenizer(nlp_de.vocab)

#from spacy.lemmatizer import Lemmatizer
#from spacy.lookups import Lookups
#lookups = Lookups()
#lemmatizer = Lemmatizer(lookups)
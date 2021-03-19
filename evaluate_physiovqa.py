#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:49:43 2020

@author: scro3517
"""

import torch
import numpy as np
from prepare_miscellaneous import obtain_information, obtain_saved_weights_name, make_saving_directory, modify_dataset_order_for_multi_task_learning, obtain_load_path_dir, determine_classification_setting
import itertools
import argparse
import pickle
import wfdb
from prepare_brazil_ecg import load_annotations_and_modify_labels

from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from operator import itemgetter
from spacy_pipeline_101 import obtain_token_mapping

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
from openTSNE import TSNE

tsne = TSNE(
    perplexity=10,
    initialization="pca",
    metric="euclidean",
    n_jobs=8,
    random_state=3,
)

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                           AutoMinorLocator)

plt.style.use('seaborn-darkgrid')
sns.set(font_scale=2)
#%%

dataset_list = ['physionet','physionet2017','cardiology','ptb','fetal','physionet2016','physionet2020','chapman','chapman_pvc','ptbxl','mimic','brazil']#,'cipa']
batch_size_list = [256, 256, 16, 64, 64, 256, 256, 256, 256, 128, 512, 16] #128 ptbxl supervised #64 for ptbxl MARGE else 128
lr_list = [1e-4, 1e-4, 1e-4, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 1e-3, 1e-3, 1e-4] #1e-5 ptbxl supervised #1e-4 for PTBXL MARGE else 1e-3
nleads = 12 # 12 | 4
if nleads == 12:
    leads_list = [None,None,None,'i','Abdomen 1','i',"['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']",
                  "['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']","['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']",
                  "['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']","['All']","['All']"] #'II' for one lead, ['II','V1',etc.] for more leads
elif nleads == 4:
    leads_list = [None,None,None,'i','Abdomen 1','i',"['II', 'V2', 'aVL', 'aVR']","['II', 'V2', 'aVL', 'aVR']","['II', 'V2', 'aVL', 'aVR']","['II', 'V2', 'AVL', 'AVR']"] #'II' for one lead, ['II','V1',etc.] for more leads
class_pair = ['','','','','','','','','','','',''] #3rd from last should be 'All Terms' when using traditional chapman loading

input_type = 'multi' #options: 'single' | 'multi' (only needed for ptbxl so far)
output_type = 'single' #options: 'single' | 'multi' (only needed for ptbxl so far) #single label or multiple labels per ecg recording
data2bs_dict = dict(zip(dataset_list,batch_size_list))
data2lr_dict = dict(zip(dataset_list,lr_list))
data2leads_dict = dict(zip(dataset_list,leads_list))
data2classpair_dict = dict(zip(dataset_list,class_pair))

""" Not Used - Just for User to Know """
perturbation_options = ['Gaussian','FlipAlongY','FlipAlongX','SpecAugment']
downstream_task_options = ['','contrastive_ms','contrastive_ml','contrastive_msml','obtain_representation_contrastive'] #load normal data, load patient data for CPPC, load patient data for CPPC
""" ------------ """
                
trials_to_run_dict = {
                'CMC':
                {'downstream_task':'contrastive_ss',
                 'nencoders':2, #must be same as nviews as per paper by Isola
                 'nviews':2}, #determines number of perturbations to perform
                'SimCLR':
                {'downstream_task':'contrastive_ss',
                 'nencoders':1, #this can be changed independently of nviews
                 'nviews':2}, #default method only contains 2 views
                'BYOL':
                {'downstream_task':'contrastive_ss',
                 'nencoders':1, #this can be changed independently of nviews
                 'nviews':2}, 
                'CMSC':
                {'downstream_task':'contrastive_ms', #determines which dataset version to load
                 'nencoders':1, #this can be changed independently of nviews
                 'nviews':2}, #changing this will require remaking dataset #nviews = nsegments 
                'CMLC':
                {'downstream_task':'contrastive_ml', #determines which dataset version to load
                 'nencoders':1, #this can be changed independently of nviews
                 'nviews':nleads}, #changing this will require remaking dataset #nviews = nleads
                'CMSMLC':
                {'downstream_task':'contrastive_msml', #determines which dataset version to load
                 'nencoders':1, #this can be changed independently of nviews
                 'nviews':nleads}, #changing this will require remaking dataset #nviews = nleads * nsegments
                'Linear':
                {'downstream_task':'contrastive_ss', #load ordinary datasets
                 'nencoders':1, #this will depend on original self-supervision method used e.g. CPPC, CMC, etc.  
                 'nviews':1}, #changing this will require remaking dataset 
                'Fine-Tuning':
                {'downstream_task':'contrastive_ss', #load ordinary datasets
                 'nencoders':1, #this will depend on original self-supervision method used e.g. CPPC, CMC, etc.  
                 'nviews':1}, #changing this will require remaking dataset 
                'Random':
                {'downstream_task':'contrastive_ss', #load ordinary datasets
                 'nencoders':1, #this will depend on original self-supervision method used e.g. CPPC, CMC, etc.  
                 'nviews':1}, #changing this will require remaking dataset 
                'AutoEnc':
                {'downstream_task':'contrastive_ss', #load ordinary datasets
                 'nencoders':1, #this will depend on original self-supervision method used e.g. CPPC, CMC, etc.  
                 'nviews':1}, #changing this will require remaking dataset 
                }    

#%%
def run_configurations(basepath_to_data,phases,input_perturbed,perturbation,second_input_perturbed,second_perturbation,trial_to_load_list,trial_to_run_list,BYOL_tau_list,embedding_dim_list,downstream_dataset_list,second_dataset_list,labelled_fraction_list,band_types=['frequency','time'],nbands=[1,1],band_width=[0.1,0.1],
                       output_type='single',input_type='single',encoder_type='cnn_small',decoder_type_list=['lstm'],num_layers_list=[1],attn=False,auxiliary_loss=False,attn_loss=False,goal_list=['IC'],pretrained_embeddings=False,freeze_embeddings=False,pretrained_encoder=False,freeze_encoder=False,pretrained_decoder=False,freeze_decoder=False,
                       dest_lang_nested_list=['en'],target_token_selection_list=['uniform'],replacement_prob_list=[0.15],
                       obtain_metrics_df=False,plot_token_distribution=False,plot_embeddings=False,obtain_sample_sentences=False,
                       top=True,bottom=False,k=1,approach='pca',perform_retrieval=False,
                       retrieve_signal_and_tokens=False,
                       retrieve_cross_lingual_tokens=False,
                       query_token='normal',query_lang='en',
                       plot_pretraining_embeddings=False,
                       plot_signal_and_sentences=False,
                       plot_diversity=False,
                       plot_more_with_less=False,
                       plot_learning_curves=False):
    """ Run All Experiments 
    Args:
        phases (list): list of phases for training        
        trial_to_load_list (list): list of trials to load #this is needed for fine-tuning later on
        trial_to_run_list (list): list of trials to run
        embedding_dim_list (list): size of embedding for representation
        downstream_dataset_list (list): list of datasets to perform experiments on
    """
    #band_types=band_types,nbands=nbands,band_width=band_width,
    metrics_df = pd.DataFrame()
    for dest_lang_list in dest_lang_nested_list:
        for trial_to_load,trial_to_run in zip(trial_to_load_list,trial_to_run_list):
            for BYOL_tau in BYOL_tau_list: #will consist of one entry if performing other experiments
                for embedding_dim in embedding_dim_list: #embedding dimension to use for pretraining
                    for downstream_dataset in downstream_dataset_list: #dataset used for pretraining
                        for second_dataset in second_dataset_list: #dataset used for evaluation down the line
                            for labelled_fraction in labelled_fraction_list:
                                
                                """ Iterate Over Decoder Types (LSTM or Transformer) """
                                for decoder_type in decoder_type_list:
                                    """ Iterate Over Number of Layers in Decoder (LSTM or Transformer) """
                                    for num_layers in num_layers_list:
                                        """ Iterate Over Pre-training Tasks """
                                        for goal in goal_list:
                                            
                                            if trial_to_load in ['Language_Detection','Language_Change_Detection','MLM','ELECTRA']:
                                                curr_replacement_prob_list = replacement_prob_list
                                                curr_target_token_selection_list = target_token_selection_list
                                            else:
                                                curr_replacement_prob_list = ['NA']
                                                curr_target_token_selection_list = ['NA']
                                            
                                            """ Iterate Over Token Replacement Probability """
                                            for replacement_prob in curr_replacement_prob_list:
                                                """ Iterate Over Target Token Selection Strategy """
                                                for target_token_selection in curr_target_token_selection_list:
    
                                                    """ ******* Activate These For Linear of Fine-Tuning Purposes ******* """
    
                                                    """ Encoder Controls """
                                                    current_pretrained_encoder = {'dataset': downstream_dataset, 'supervision': 'Supervised'} if pretrained_encoder == True else False #options: False | Dict (loads pretrained parameters)
                                                    current_freeze_encoder = freeze_encoder #only applies when pretrained_encoder is not False 
                                                    """ Decoder Controls """
                                                    current_pretrained_decoder = {'dataset': downstream_dataset, 'supervision': trial_to_load, 'langs': dest_lang_list, 'target_token_selection': target_token_selection, 'replacement_prob': replacement_prob, 'num_layers': num_layers} if pretrained_decoder == True else False #options: False | Dict (loads pretrained parameters)
                                                    current_freeze_decoder = freeze_decoder
                                                    """ Word Embeddings Controls """
                                                    current_pretrained_embeddings = {'dataset': downstream_dataset, 'supervision': trial_to_load, 'langs': dest_lang_list, 'target_token_selection': target_token_selection, 'replacement_prob': replacement_prob, 'num_layers': num_layers} if pretrained_embeddings == True else False #options: True | False (should you load pre-trained embeddings - currently this is only based on Spacy embeddings)
                                                    current_freeze_embeddings = freeze_embeddings
    
                                                    """ End """
    
                                                    downstream_task, nencoders, nviews = trials_to_run_dict[trial_to_run].values()
                                                    #input_perturbed, perturbation = trials_to_load_dict[trial_to_load].values()
                                                    saved_weights = obtain_saved_weights_name(trial_to_run,trial_to_load,phases)
                                                    
                                                    """ Information for save_path_dir """
                                                    original_leads, original_batch_size, original_held_out_lr, original_class_pair, original_modalities, original_fraction = obtain_information(trial_to_load,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict,extra_trial=trial_to_run) #should be trial_to_load alone
                                                    """ Information for actual training --- trial_to_run == trial_to_load when pretraining so they are the same """
                                                    leads, batch_size, held_out_lr, class_pair, modalities, fraction = obtain_information(trial_to_run,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict) #should be trial_to_run alone
                                                    
                                                    max_epochs = 25 #hard stop for training
                                                    max_seed = 5
                                                    seeds = np.arange(0,max_seed)#max_seed)
    
                                                    """ Obtain Save Path Dirs For All Seeds - We Can Choose Later """
                                                    all_dirs = []
                                                    for seed in seeds:
                                                        save_path_dir, seed = make_saving_directory(phases,downstream_dataset,trial_to_load,trial_to_run,seed,max_seed,downstream_task,BYOL_tau,embedding_dim,original_leads,input_perturbed,perturbation,band_types,nbands,band_width,encoder_type,decoder_type,attn,attn_loss,goal,
                                                                                                    current_pretrained_embeddings,current_freeze_embeddings,current_pretrained_encoder,current_freeze_encoder,
                                                                                                    current_pretrained_decoder,current_freeze_decoder,dest_lang_list,target_token_selection,
                                                                                                    replacement_prob,num_layers)
                            
                                                        if trial_to_run in ['Linear','Fine-Tuning','Random']:  
                                                            original_downstream_dataset,modalities,leads,class_pair,fraction = modify_dataset_order_for_multi_task_learning(second_dataset,modalities,leads,class_pair,fraction)
                                                        else:
                                                            original_downstream_dataset = downstream_dataset #to avoid overwriting downstream_dataset which is needed for next iterations
                                                        
                                                        load_path_dir, save_path_dir = obtain_load_path_dir(phases,save_path_dir,trial_to_load,trial_to_run,second_input_perturbed,second_perturbation,second_dataset,labelled_fraction,leads,max_seed,downstream_task,evaluation=True)
                                                        if save_path_dir in ['do not train','do not test']:
                                                            continue
                                                        
                                                        all_dirs.append(save_path_dir)

                                                    """ BEGIN EVALUATION PROCEDURES """
    
                                                    """ ****** A ****** Obtain Average Metric Performance Across Seeds """
                                                    if obtain_metrics_df == True:
                                                        """ Obtain DF With Results For Single Method """
                                                        single_method_metrics_df = load_metrics_all_seeds(all_dirs,seeds,trial_to_load)
                                                        """ Combined Results Across Methods """
                                                        metrics_df = pd.concat((metrics_df,single_method_metrics_df),0)
                                                        #return metrics_df
                                                                                                    
                                                    """ ****** B ****** Obtain Distribution of Target and Output Sentence Tokens """
                                                    if plot_token_distribution == True:
                                                        save_path_dir = all_dirs[0]
                                                        target_sentences = load_target_sentences(save_path_dir)
                                                        output_sentences = load_output_sentences(save_path_dir)
                                                        plot_most_frequent_tokens(target_sentences,save_path_dir,dest_lang_list=dest_lang_list,k=k,palette='Blues_d',ground_truth=True)
                                                        plot_most_frequent_tokens(output_sentences,save_path_dir,dest_lang_list=dest_lang_list,k=k,palette='Oranges_d',ground_truth=False)
    
                                                    """ ****** C ****** Project Embeddings and Plot Them As Scatter Plot """                                                    
                                                    if plot_embeddings == True:
                                                        save_path_dir = all_dirs[0]
                                                        plot_token_embeddings(save_path_dir,downstream_dataset,dest_lang_list,approach=approach)
                                                        
                                                    """ ****** D ****** Obtain Sample Sentences (Target and Predicted) From Each Language """                                                    
                                                    if obtain_sample_sentences == True:
                                                        save_path_dir = all_dirs[0]
                                                        target_sentences = load_target_sentences(save_path_dir)
                                                        output_sentences = load_output_sentences(save_path_dir)
                                                        tgt_sentences, out_sentences, indices, scores = obtain_sample_sentences_from_all_languages(target_sentences,output_sentences,top=top,bottom=bottom,k=k)
                                                        #return tgt_sentences, out_sentences, indices, scores
                                                    
                                                    """ ****** E ****** Project Self-supervised Embeddings and Plot Them As Scatter Plot """
                                                    if plot_pretraining_embeddings == True:
                                                        pretraining_path_dir = obtain_pretraining_save_path_dir(save_path_dir)
                                                        plot_token_embeddings(pretraining_path_dir,downstream_dataset,dest_lang_list,approach=approach)
                                                    
                                                    """ ****** F ****** Perform Cross-Lingual Token Retrieval Using Embedding Dict """                                                    
                                                    if retrieve_cross_lingual_tokens == True:
                                                        save_path_dir = all_dirs[0]
                                                        plot_cross_lingual_tokens(save_path_dir,downstream_dataset,query_token,query_lang,dest_lang_list,k)
                                                    
                                                    """ ****** G ****** Perform Token-Based Signal Retrieval and Signal-Based Token Retrieval """
                                                    if retrieve_signal_and_tokens == True:
                                                        save_path_dir = all_dirs[0]
                                                        plot_tokens_and_signals(save_path_dir,downstream_dataset,query_token,query_lang,dest_lang_list,k)
    
                                                    """ ****** H ****** Plot Signal and Corresponding Multi-lingual Sentences """
                                                    if plot_signal_and_sentences == True:
                                                        save_path_dir = all_dirs[0]
                                                        target_sentences = load_target_sentences(save_path_dir)
                                                        output_sentences = load_output_sentences(save_path_dir)
                                                        tgt_sentences, out_sentences, indices, scores = plot_signal_and_sentences_from_all_languages(target_sentences,output_sentences,downstream_dataset,top=top,bottom=bottom,k=k)
    
                                                    """ ****** I ****** Plot Diversity of Generated Text """
                                                    if plot_diversity == True:
                                                        curr_metrics_df = calculate_diversity_all_seeds(all_dirs,trial_to_load)
                                                        metrics_df = pd.concat((metrics_df,curr_metrics_df),0)
                                                        
                                                    """ ****** J ****** Plot Effect of Labelled Fraction on Multiple Methods """
                                                    if plot_more_with_less == True:
                                                        """ Obtain DF With Results For Single Method """
                                                        single_method_metrics_df = load_metrics_all_seeds(all_dirs,seeds,trial_to_load)
                                                        single_method_metrics_df['Fraction'] = pd.DataFrame([labelled_fraction]*single_method_metrics_df.shape[0],index=single_method_metrics_df.index)
                                                        """ Combined Results Across Methods """
                                                        metrics_df = pd.concat((metrics_df,single_method_metrics_df),0)
                                                    
                                                    """ ****** K ****** Plot (Line) Learning Curves of Multiple Methods """                                                    
                                                    if plot_learning_curves == True:
                                                        """ Obtain DF With Results For Single Method """
                                                        single_method_metrics_df = load_metrics_all(all_dirs,seeds,trial_to_load)
                                                        """ Combined Results Across Methods """
                                                        metrics_df = pd.concat((metrics_df,single_method_metrics_df),0)                                                        

    if obtain_metrics_df == True:
        return metrics_df
    elif plot_learning_curves == True:
        plot_curves(metrics_df)
        return metrics_df
    elif obtain_sample_sentences == True or plot_signal_and_sentences == True:
        return tgt_sentences, out_sentences, indices, scores
    elif plot_diversity == True:
        plot_bar_diversity(metrics_df)
        return metrics_df
    elif plot_more_with_less == True:
        plot_effect_of_labelled_fraction(metrics_df)
        return metrics_df

#%%
def calc_dist_between_tokens(save_path_dir,dataset_name,dest_lang_list,k):    
    """ Load Mapping from IDs to Tokens """
    token2id_dict, df = obtain_token_mapping(dataset_name,dest_lang_list)
    id2token_dict = {lang: {id_entry:token for token,id_entry in lang_token2id_dict.items()} for lang,lang_token2id_dict in token2id_dict.items()}
    """ Load Embeddings """
    text_embeds = load_embeddings(save_path_dir)
        
    id2id = dict()
    for lang,id2embeds in text_embeds.items():
        lang_embeds = torch.stack(list(id2embeds.values())).detach().cpu().numpy() # N x 300
        id2id[lang] = dict()
        for other_lang,other_id2embeds in text_embeds.items():
            """ Comment to Compare to Similar Language Also """
            #if other_lang != lang:
            other_lang_embeds = torch.stack(list(other_id2embeds.values())).detach().cpu().numpy() # P x 300                
            """ Obtain Similarity Matrix Between Language Tokens and Other Language Representations """
            sim_matrix = np.matmul(lang_embeds,other_lang_embeds.T)
            m,n = sim_matrix.shape
            """ Normalize Matrix """
            norm_lang = np.tile(np.expand_dims(np.linalg.norm(lang_embeds,axis=1),1), (1,n))
            norm_other_lang = np.tile(np.expand_dims(np.linalg.norm(other_lang_embeds,axis=1),0), (m,1))
            norm_matrix = norm_lang * norm_other_lang
            sim_matrix = sim_matrix/norm_matrix
            """ Obtain Nearest K Elements To Query = Current Lang Tokens """
            nearest_tokens = np.fliplr(sim_matrix.argsort(axis=1))[:,:k] # N x K
            """ Store Info """
            id2id[lang][other_lang] = dict(zip(list(id2embeds.keys()),nearest_tokens))
    
    return id2id, id2token_dict, token2id_dict, df

def plot_cross_lingual_tokens(save_path_dir,dataset_name,query_token,query_lang,dest_lang_list,k):
    """ Calculate Distance Between Tokens """
    id2id, id2token_dict, token2id_dict, df = calc_dist_between_tokens(save_path_dir,dataset_name,dest_lang_list,k)
    
    """ Obtain Nearest Signal to Query """
    qtoken = query_token #'normal' #'pathologic', 'normal', 'depression', 'bradycardia' # query_token
    qid = token2id_dict[query_lang][qtoken]
    
    """ Retrieve Nearest Tokens From Other Langs """
    nearest_tokens = dict()
    for other_lang in dest_lang_list:
        """ Comment to Compare to Similar Language Also """
        #if other_lang != query_lang:
        nearest_token_indices = id2id[query_lang][other_lang][qid]
        nearest_lang_tokens = [id2token_dict[other_lang][index] for index in nearest_token_indices]
        nearest_tokens[other_lang] = nearest_lang_tokens
    
    """ Plot Text From Each Language """
    fig,axes = plt.subplots(len(dest_lang_list),1,figsize=(20,12))
    for i,(lang,token_list) in enumerate(nearest_tokens.items()):
        axes[i].text(0,0.5,[lang.upper()] + token_list)    
    
#%%
""" Function to Perform Retrieval """

def calc_dist_between_tokens_and_signals(save_path_dir,dataset_name,dest_lang_list,k):    
    """ Load Mapping from IDs to Tokens """
    token2id_dict, df = obtain_token_mapping(dataset_name,dest_lang_list)
    id2token_dict = {lang: {id_entry:token for token,id_entry in lang_token2id_dict.items()} for lang,lang_token2id_dict in token2id_dict.items()}
    """ Load Embeddings """
    text_embeds = load_embeddings(save_path_dir)
    
    """ Obtain Path to Signal Representations """
    save_path_dir_split = save_path_dir.split('/')
    prefix = '/'.join(save_path_dir_split[:6])
    embedding = save_path_dir_split[7]
    signal_path = os.path.join(prefix,embedding,'Supervised/seed0',dataset_name,'training_fraction_1.00')
    """ Load Representations in Validation Set """
    sig_embeds = torch.load(os.path.join(signal_path,'representations')) # M x 300
    
    id2sig = dict()
    sig2id = dict()
    for lang,id2embeds in text_embeds.items():
        lang_embeds = torch.stack(list(id2embeds.values())).detach().cpu().numpy() # N x 300
        """ Obtain Similarity Matrix Between Language Tokens and Signal Representations """
        sim_matrix = np.matmul(lang_embeds,sig_embeds.T)
        m,n = sim_matrix.shape
        """ Normalize Matrix """
        norm_lang = np.tile(np.expand_dims(np.linalg.norm(lang_embeds,axis=1),1), (1,n))
        norm_sigs = np.tile(np.expand_dims(np.linalg.norm(sig_embeds,axis=1),0), (m,1))
        norm_matrix = norm_lang * norm_sigs
        sim_matrix = sim_matrix/norm_matrix
        """ Obtain Nearest K Elements To Query = Tokens """
        nearest_sigs = np.fliplr(sim_matrix.argsort(axis=1))[:,:k] # N x K
        """ Obtain Nearest K Elements to Query = Signals """
        nearest_tokens = np.flipud(sim_matrix.argsort(axis=0))[:k,:].T # M x K
        """ Store Info """
        id2sig[lang] = dict(zip(list(id2embeds.keys()),nearest_sigs))
        sig2id[lang] = dict(zip(np.arange(len(sig_embeds)),nearest_tokens))
    
    return id2sig, sig2id, id2token_dict, token2id_dict, sig_embeds, df

def load_actual_signals(dataset_name,input_type='multi',output_type='single',basepath='/mnt/SecondaryHDD',goal=''):
    
    if dataset_name in ['ptbxl']:
        loadpath = os.path.join(basepath,'PTB-XL','patient_data','%s_input' % input_type,'%s_output' % output_type)
        with open(os.path.join(loadpath,'phase_to_paths.pkl'),'rb') as f:
            phase_to_paths = pickle.load(f)
        with open(os.path.join(loadpath,'phase_to_leads.pkl'),'rb') as f:
            phase_to_leads = pickle.load(f) #not used right now
        with open(os.path.join(loadpath,'phase_to_segment_number.pkl'),'rb') as f:
            phase_to_segment_number = pickle.load(f)
#    elif dataset_name in ['chapman']:
#        loadpath = os.path.join(basepath,'chapman_ecg','patient_data',goal,'%s_input' % input_type,'%s_output' % output_type)
#        with open(os.path.join(loadpath,'phase_to_paths.pkl'),'rb') as f:
#            phase_to_paths = pickle.load(f)
#        with open(os.path.join(loadpath,'phase_to_leads.pkl'),'rb') as f:
#            phase_to_leads = pickle.load(f) #not used right now
#        with open(os.path.join(loadpath,'phase_to_segment_number.pkl'),'rb') as f:
#            phase_to_segment_number = pickle.load(f)
#        with open(os.path.join(loadpath,'phase_to_questions.pkl'),'rb') as f:
#            phase_to_questions = pickle.load(f)
#    elif dataset_name in ['brazil']:
#        loadpath = os.path.join(basepath,'Brazil_ECG','patient_data')
#        with open(os.path.join(loadpath,'year_to_paths.pkl'),'rb') as f:
#            year_to_paths = pickle.load(f)
    
    return phase_to_paths

def plot_tokens_and_signals(save_path_dir,dataset_name,query_token,query_lang,dest_lang_list,k):
    """ Calculate Distance Between Visual and Textual Representations """
    id2sig, sig2id, id2token_dict, token2id_dict, sig_embeds, df = calc_dist_between_tokens_and_signals(save_path_dir,dataset_name,dest_lang_list,k)
    
    """ Obtain Nearest Signal to Query """
    qtoken = query_token  #'pathologic', 'normal', 'depression', 'bradycardia' #query_token
    qid = token2id_dict[query_lang][qtoken]
    nearest_sigs_indices = id2sig[query_lang][qid]
    
    """ ChoOse Nearest Signal [0] """
    nearest_sig_index = nearest_sigs_indices[0]
    """ Load Actual Signal (12-Lead) """
    phase_to_paths = load_actual_signals(dataset_name)
    path = phase_to_paths['val'][nearest_sig_index]
    label = obtain_ecg_label(path,df,dataset_name)
    print(label)
    plot_signal(path,dataset_name)
    
    """ Obtain Nearest Tokens to Signal """
    nearest_tokens = dict()
    for lang in dest_lang_list:
        nearest_lang_ids = sig2id[lang][nearest_sig_index] # K
        nearest_tokens[lang] = [id2token_dict[lang][id_entry] for id_entry in nearest_lang_ids] # K
    
    """ Plot Text From Each Language """
    fig,axes = plt.subplots(len(dest_lang_list),1,figsize=(20,12))
    for i,(lang,token_list) in enumerate(nearest_tokens.items()):
        axes[i].text(0,0.5,token_list)

def obtain_ecg_label(path,df,dataset_name,output_type='single'):
    
    if 'ptbxl' in dataset_name:
        ecg_id = int(path.split('/')[-1].split('_')[0])
        if output_type == 'single':
            label = df[df.index == ecg_id].superdiagnostic.iloc[0]
    elif 'brazil' in dataset_name:
        basepath = '/mnt/SecondaryHDD/Brazil_ECG'
        annot_df = load_annotations_and_modify_labels(basepath)
        for year,paths in path.items():
            """ Obtain Paths """
            current_phase_paths = paths['val']
            id_entry = list(map(lambda path:int(path.split('/')[-1].split('_')[0].split('TNMG')[1]),current_phase_paths))
            label = annot_df[annot_df['id_exam'] == id_entry].single_label.item()
    
    return label

def plot_signal(path,dataset_name):
    info = wfdb.rdsamp(path)
    signals = info[0]
    leads = info[1]['sig_name']    
    desired_leads = leads
    
    fig,axes = plt.subplots(3,4,figsize=(12,8))
    if dataset_name == 'chapman':
        width = 500
        final_time = width/250 #2500/5000 #chapman case with contrastive_ss formulation b/c of resampling
        #skip_factor = 1
    elif dataset_name == 'ptbxl':
        width = 2000 #1000
        final_time = width/500
        #skip_factor = 1
    elif dataset_name == 'physionet2020':
        width = 1000
        final_time = width/500
        #skip_factor = 2

    s = 0
    for c in range(4):
        for r in range(3):
            lead_index = np.where(np.in1d(leads,[desired_leads[s]]))[0][0]
            segment = signals[:,lead_index]
            """ Normalize Segment """
            segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
            """ Take Central 500 Samples for Illustration Purposes """
            subsegment = segment[len(segment)//2 - width//2:len(segment)//2 + width//2]
            lead = desired_leads[s]
            axes[r,c].plot(np.linspace(0,final_time,width),subsegment,label=lead,lw=3) #color='#1f77b4' #ax =  to get color
            activate_ecg_background(axes[r,c])
            leg = axes[r,c].legend(loc='upper right',handlelength=0, handletextpad=0,fontsize=20,frameon=True,facecolor='white',framealpha=0.75)
            #for item in leg.legendHandles:
            #    item.set_visible(True)
            s += 1
    fig.tight_layout()  

def activate_ecg_background(ax):
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter(''))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter(''))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(True,which='major',color='k',lw=0.2)
    ax.grid(True,which='minor',color='k',lw=0.05)

    ax.tick_params(left=False, bottom=False)
    ax.set_facecolor('salmon')
    ax.patch.set_alpha(0.3) 

#%%
""" Functions to Load Metrics in DataFrame Format """    

def load_metrics_single_seed(save_path_dir,trial_to_load,metric_names=['epoch_bleu','epoch_rouge','epoch_meteor']):
    single_seed_metrics_df = pd.DataFrame()
    metrics = torch.load(os.path.join(save_path_dir,'train_val_metrics_dict'))
    """ Identify Epoch From Which To Extract Results (Based on Max Bleu Score) """
    mean_bleus = list(map(lambda x:np.mean(list(x.values())),metrics['epoch_bleu']['val']))
    best_epoch = np.argmax(mean_bleus)
    
    for metric_name in metric_names:
        best_metrics = metrics[metric_name]['val'][best_epoch]
        best_metrics_df = pd.DataFrame.from_dict(best_metrics,orient='index')
        best_metrics_df['Metric'] = [metric_name] * len(best_metrics)
        best_metrics_df['Method'] = [trial_to_load] * len(best_metrics)
        single_seed_metrics_df = pd.concat((single_seed_metrics_df,best_metrics_df),0)

    single_seed_metrics_df['Lang'] = pd.DataFrame(single_seed_metrics_df.index.tolist(),index=single_seed_metrics_df.index)
    single_seed_metrics_df.index = np.arange(single_seed_metrics_df.shape[0])
    single_seed_metrics_df.columns = ['Value','Metric','Method','Lang']
    
    return single_seed_metrics_df

def load_metrics_all_seeds(all_dirs,seeds,trial_to_load,metric_names=['epoch_bleu','epoch_rouge','epoch_meteor']):
    metrics_df = pd.DataFrame()
    for save_path_dir in all_dirs:
        """ Load DataFrame With Results from Single Seed """
        single_seed_metrics_df = load_metrics_single_seed(save_path_dir,trial_to_load,metric_names=metric_names)
        """ Combine Results """
        metrics_df = pd.concat((metrics_df,single_seed_metrics_df),0)
    return metrics_df

#%%
def load_metrics_single(save_path_dir,trial_to_load,metric_names=['epoch_bleu','epoch_rouge','epoch_meteor']):
    #single_seed_metrics_df = pd.DataFrame()
    metrics = torch.load(os.path.join(save_path_dir,'train_val_metrics_dict'))
    """ Identify Epoch From Which To Extract Results (Based on Max Bleu Score) """
    
    values = []
    for metric_name in metric_names:
        for epoch,results in enumerate(metrics[metric_name]['val']):
            for lang,val in results.items():
                values.append([epoch,lang,trial_to_load,metric_name,val])
            
    values_df = pd.DataFrame(values)
    values_df.columns = ['Epoch','Lang','Method','Metric','Value']
    
    return values_df

def load_metrics_all(all_dirs,seeds,trial_to_load,metric_names=['epoch_bleu','epoch_rouge','epoch_meteor']):
    metrics_df = pd.DataFrame()
    for save_path_dir in all_dirs:
        """ Load DataFrame With Results from Single Seed """
        single_seed_metrics_df = load_metrics_single(save_path_dir,trial_to_load,metric_names=metric_names)
        """ Combine Results """
        metrics_df = pd.concat((metrics_df,single_seed_metrics_df),0)
    return metrics_df

def plot_curves(metrics_df):
    metrics_df['Method'] = metrics_df['Method'].replace({'Language_Detection':'RTLP','Language_Change_Detection':'RTLD'})
    palette_map = {'MLM':sns.color_palette('Greens_d',10)[2],'RTLP':sns.color_palette('Oranges_d',10)[2],'ELECTRA':'yellow','MARGE':'royalblue'}
    fig,axes = plt.subplots(2,1,figsize=(20,8),gridspec_kw={'height_ratios': [0.05, 0.95]})
    lineplot = sns.lineplot(x='Epoch',y='Value',hue='Method',palette=palette_map,data=metrics_df[metrics_df['Metric'] == 'epoch_bleu'])
    axes[0].remove()
    lineplot.set(xlabel='Epoch')
    lineplot.set(ylabel='BLEU')
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.2),ncol=len(metrics_df.Method.value_counts()),markerscale=5)

#%%
""" Functions to Plot Distributions of Tokens """

def load_target_sentences(save_path_dir):
    sentences = torch.load(os.path.join(save_path_dir,'target_sentences'))
    return sentences

def load_output_sentences(save_path_dir):
    sentences = torch.load(os.path.join(save_path_dir,'output_sentences'))
    return sentences    

def plot_most_frequent_tokens(sentences,save_path_dir,dest_lang_list,k,palette,ground_truth):
#    fig,axes = plt.subplots(len(dest_lang_list),1,figsize=(20,8),gridspec_kw={'height_ratios': [1]*len(dest_lang_list)},
#                            sharex='all',sharey='all')    
    sns.set(font_scale=3)
    for i,lang in enumerate(dest_lang_list):
        #ax_count = 2*i
        """ Obtain Language Specific Text """
        txt = sentences[lang]
        """ Flatten Text """
        entries = []
        for sentence in txt:
            entries.extend(sentence)
        """ Obtain Frequency of Tokens """ 
        counts_dict = dict(Counter(entries))
        """ Convert Counts Dict to Pandas DF """
        counts_df = pd.DataFrame.from_dict(counts_dict,orient='index')
        counts_df['Token'] = pd.DataFrame(counts_df.index.tolist(),index=counts_df.index)
        counts_df.columns = ['Counts','Tokens']
        """ Plot k Most Frequent Tokens """
        fig,axes = plt.subplots(2,1,figsize=(20,8),gridspec_kw={'height_ratios': [0.8, 0.2]})
        barplot = sns.barplot(x='Tokens',y='Counts',palette=palette,data=counts_df.sort_values(by='Counts',ascending=False).iloc[:k,:],ax=axes[0])
        axes[0].set_ylim([0,3000])
        axes[1].remove()
        
        for item in barplot.get_xticklabels():
            item.set_rotation(45)
            
        barplot.set(xlabel=None)
        barplot.set(ylabel=None)
        barplot.set_yticklabels([])
        
        """ Save Figure """
        savepath = '/mnt/SecondaryHDD/PhysioVQA Results/figures'
        #method = save_path_dir.split('/')[0].split('_')[3] if ground_truth == False else ''
        suffix = 'target_distribution_%s' % (lang) if ground_truth == True else 'output_distribution_%s' % (lang)
        #final_path = os.path.join(os.path.join(savepath,method,suffix))
        #if os.path.exists(final_path) == False:
        #    os.makedirs(final_path) 
        #plt.savefig(os.path.join(os.path.join(savepath,suffix)))
        #plt.close()
        
#%%
""" Functions to Plot Projections of Embeddings """
    
def load_embeddings(save_path_dir):
    embeddings = torch.load(os.path.join(save_path_dir,'id2embedding_dict'))
    return embeddings

def project_embeddings(embeddings,id2token_dict,approach='pca'):
    all_embeds_df = pd.DataFrame()
    all_auxiliary_df = pd.DataFrame()
    for lang,id2embeds in embeddings.items():
        """ Obtain Tokens from Ids """
        id_entries = list(id2embeds.keys())
        tokens = list(itemgetter(*id_entries)(id2token_dict[lang]))
        embeds_list = []
        """ Iterate Over Language Token Embeddings and Convert to Numpy """
        for id_entry,embed in id2embeds.items():
            embeds_list.append(embed.detach().cpu().numpy())
        """ Convert Embeddings into DataFrame Format """
        embeds_df = pd.DataFrame(np.array(embeds_list),index=tokens)
        auxiliary_df = pd.DataFrame([lang] * embeds_df.shape[0],index=tokens,columns=['Lang'])
        auxiliary_df['Token'] = pd.DataFrame(tokens,index=tokens)
        """ Combine Data """
        all_embeds_df = pd.concat((all_embeds_df,embeds_df),0)
        all_auxiliary_df = pd.concat((all_auxiliary_df,auxiliary_df),0)
    
    """ Perform Projection """
    print('Performing Projection...')
    all_embeds_np = all_embeds_df.to_numpy()
    if approach == 'tsne':
        embeds_2d = tsne.fit(all_embeds_np)
    elif approach == 'pca':
        embeds_2d = pca.fit_transform(all_embeds_np)
    
    all_tokens = all_embeds_df.index.tolist()
    proj_df = pd.DataFrame(embeds_2d,index=all_tokens,columns=['Axis 1','Axis 2'])
    proj_df['Token'] = pd.DataFrame(all_tokens,index=proj_df.index)
    proj_df['Lang'] = all_auxiliary_df.Lang
    
    return proj_df
        
def plot_token_embeddings(save_path_dir,dataset_name,dest_lang_list,approach='pca'):
    
    """ Load Mapping from IDs to Tokens """
    token2id_dict, _ = obtain_token_mapping(dataset_name,dest_lang_list)
    id2token_dict = {lang: {id_entry:token for token,id_entry in lang_token2id_dict.items()} for lang,lang_token2id_dict in token2id_dict.items()}
    """ Load Embeddings """
    embeddings = load_embeddings(save_path_dir)
    """ Perform Projection of Embeddings """
    proj_df = project_embeddings(embeddings,id2token_dict,approach=approach)
    """ Plot Projections """
    fig,ax = plt.subplots(figsize=(20,8))
    sns.scatterplot(x='Axis 1',y='Axis 2',hue='Lang',data=proj_df,palette='deep',ax=ax)

def obtain_pretraining_save_path_dir(save_path_dir):    
    save_path_dir_split = save_path_dir.split('/')
    prefix = '/'.join(save_path_dir_split[:5])
    decoder = save_path_dir_split[6]
    embedding = save_path_dir_split[7]
    pretraining = save_path_dir_split[9]
    
    pretraining_path = pretraining.split('_')
    dataset = pretraining_path[2]
    method = pretraining_path[3]
    langs = 'langs_%s' % str(pretraining_path[4])
    sampling = 'target_token_selection_%s' % str(pretraining_path[5])
    replacement_prob = 'replacement_prob_%s' % str(pretraining_path[6])
    num_layers = 'num_layers_%s' % str(pretraining_path[7])
    
    new_path = os.path.join(prefix,decoder,num_layers,embedding,method,langs,sampling,replacement_prob,'seed0',dataset,'training_fraction_1.00')
    return new_path
    
#%%
""" Functions to Obtain Sample Sentences Based on Bleu Score """
    
def calculate_bleu(target_sentences,output_sentences):
    langs = list(target_sentences.keys())
    scores = dict()
    for lang in langs:
        bleu_scores = [sentence_bleu([tgt],out,weights=[1]) for out,tgt in zip(output_sentences[lang],target_sentences[lang])]
        scores[lang] = bleu_scores
    return scores

def calculate_meteor(target_sentences,output_sentences):
    langs = list(target_sentences.keys())
    scores = dict()
    for lang in langs:
        meteor_scores = [meteor_score([' '.join(tgt)],' '.join(out)) for out,tgt in zip(output_sentences[lang],target_sentences[lang])]
        scores[lang] = meteor_scores
    return scores

def calculate_rouge(target_sentences,output_sentences):
    langs = list(target_sentences.keys())
    scores = dict()
    scorer = rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    for lang in langs:
        """ Obtain Fmeasure From Rouge Scores """  #expects list of strings and list of strings
        rouge_scores = [scorer.score(' '.join(tgt),' '.join(out))['rougeL'][-1] for out,tgt in zip(output_sentences[lang],target_sentences[lang])]
        scores[lang] = rouge_scores
    return scores

def obtain_sample_sentences_from_all_languages(target_sentences,output_sentences,top=True,bottom=False,k=1):
    """ Calculate Scores For EACH Sentence in Validation Set """
    bleu_scores = calculate_bleu(target_sentences,output_sentences)
    meteor_scores = calculate_meteor(target_sentences,output_sentences)
    rouge_scores = calculate_rouge(target_sentences,output_sentences)
    
    tgt_sentences = dict()
    out_sentences = dict()
    all_indices = dict()
    all_scores = dict()
    for lang,scores in bleu_scores.items():
        """ Obtain Top/Bottom Performing Sentences """
        indices_sorted = sorted(np.arange(len(scores)),key=lambda i:scores[i])
        if top == True:
            selected_indices = indices_sorted[-k:]
        elif bottom == True:
            selected_indices = indices_sorted[:k]
        """ Store Sentences """
        all_indices[lang] = selected_indices
        tgt_sentences[lang] = [' '.join(target_sentences[lang][index]) for index in selected_indices]
        out_sentences[lang] = [' '.join(output_sentences[lang][index]) for index in selected_indices]
        """ Store Corresponding Scores """
        bleu_score_list = [scores[index] for index in selected_indices]
        meteor_score_list = [meteor_scores[lang][index] for index in selected_indices]
        rouge_score_list = [rouge_scores[lang][index] for index in selected_indices]
        all_scores[lang] = {'bleu':bleu_score_list,'meteor':meteor_score_list,'rouge':rouge_score_list}
    return tgt_sentences, out_sentences, all_indices, all_scores

def plot_signal_and_sentences_from_all_languages(target_sentences,output_sentences,dataset_name,top=True,bottom=False,k=1):
    """ Calculate Scores For EACH Sentence in Validation Set """
    bleu_scores = calculate_bleu(target_sentences,output_sentences)
    meteor_scores = calculate_meteor(target_sentences,output_sentences)
    rouge_scores = calculate_rouge(target_sentences,output_sentences)
    
    tgt_sentences = dict()
    out_sentences = dict()
    all_indices = dict()
    eval_scores = dict()
    
    print('Selecting Best Index...')
    """ Obtain Scores Across Languages """
    for l,(lang,scores) in enumerate(bleu_scores.items()):
        if l == 0:
            all_scores = list(map(lambda entry:[entry], scores))
        else:
            for i,score in enumerate(scores):
                all_scores[i].append(score)
    """ Average Scores Across Languages """
    mean_scores = list(map(lambda x:np.mean(x), scores))
    """ Sort Mean Scores """
    indices_sorted = list(sorted(np.arange(len(mean_scores)), key=lambda i:mean_scores[i], reverse=True))
    """ Best Index """
    best_index = indices_sorted[0]
    selected_indices = [best_index]
    
    for lang,scores in bleu_scores.items():
        tgt_sentences[lang] = [' '.join(target_sentences[lang][index]) for index in selected_indices]
        out_sentences[lang] = [' '.join(output_sentences[lang][index]) for index in selected_indices]
        """ Store Corresponding Scores """
        bleu_score_list = [scores[index] for index in selected_indices]
        meteor_score_list = [meteor_scores[lang][index] for index in selected_indices]
        rouge_score_list = [rouge_scores[lang][index] for index in selected_indices]
        eval_scores[lang] = {'bleu':bleu_score_list,'meteor':meteor_score_list,'rouge':rouge_score_list}
    
    phase_to_paths = load_actual_signals(dataset_name)
    path = phase_to_paths['val'][best_index]
    plot_signal(path,dataset_name)
    
    return tgt_sentences, out_sentences, all_indices, eval_scores

#%%
""" Functions to Calculate Diversity of Generated Text """    

def calculate_self_bleu(current_sentences):
    """ Treat Sentence As Target and Compare to All Other Sentences """
    bleu_scores = []
    i = 0
    for sentence in tqdm(current_sentences):
        sentence_scores = [sentence_bleu([sentence],out,weights=[1]) for j,out in enumerate(current_sentences) if i != j]
        bleu_scores.extend(sentence_scores)
        i += 1
    return bleu_scores

def calculate_diversity_single_seed(output_sentences,trial_to_load,seed):
    langs = list(output_sentences.keys())
    results_df = pd.DataFrame()
    print('Calculating Self-Bleu...')
    for lang in langs:
        current_sentences = output_sentences[lang]
        """ INSERT METRIC FUNCTION HERE """
        scores = calculate_self_bleu(current_sentences)
        scores = np.mean(scores)
        """ Convert Results to DF """
        m = 1 if isinstance(scores,float) else len(scores)
        auxiliary_df = pd.DataFrame([lang]*m,index=range(m))
        auxiliary_df['Method'] = pd.DataFrame([trial_to_load]*m,index=range(m))
        auxiliary_df['Seed'] = pd.DataFrame([seed]*m,index=range(m))
        auxiliary_df['Self-Bleu'] = pd.DataFrame([scores],index=range(m)) if m == 1 else pd.DataFrame(scores,index=range(m))
        auxiliary_df.columns = ['Lang','Method','Seed','Self-Bleu']
        """ Store Results """
        results_df = pd.concat((results_df,auxiliary_df),0)
        
    return results_df

def calculate_diversity_all_seeds(all_dirs,trial_to_load):
    results_df = pd.DataFrame()
    for seed,save_path_dir in enumerate(all_dirs):
        """ Load Generated Sentences """
        output_sentences = load_output_sentences(save_path_dir)
        """ Calculate Negative of Self-Bleu """
        curr_results_df = calculate_diversity_single_seed(output_sentences,trial_to_load,seed)
        """ Store Results """
        results_df = pd.concat((results_df,curr_results_df),0)
    return results_df

def plot_bar_diversity(results_df):
    results_df['Method'] = results_df['Method'].replace({'Language_Detection':'RTLP','Language_Change_Detection':'RTLD'})
    palette_map = {'MLM':sns.color_palette('Greens_d',10)[2],'RTLP':sns.color_palette('Oranges_d',10)[2],'ELECTRA':'yellow','MARGE':'royalblue'}
    fig,axes = plt.subplots(2,1,figsize=(20,8),gridspec_kw={'height_ratios': [0.1, 0.9]})
    barplot = sns.barplot(x='Lang',y='Self-Bleu',hue='Method',edgecolor='k',palette=palette_map,data=results_df,ax=axes[1])
    axes[0].remove()
    barplot.set(xlabel=None)
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.2),ncol=len(results_df.Method.value_counts()),markerscale=5)

#%%
def plot_effect_of_labelled_fraction(metrics_df):
    metrics_df['Method'] = metrics_df['Method'].replace({'Language_Detection':'RTLP','Language_Change_Detection':'RTLD'})
    palette_map = {'MLM':sns.color_palette('Greens_d',10)[2],'RTLP':sns.color_palette('Oranges_d',10)[2],'ELECTRA':'yellow','MARGE':'royalblue'}
    fig,axes = plt.subplots(2,1,figsize=(20,8),gridspec_kw={'height_ratios': [0.05, 0.95]})
    """ Choose Metric of Interest """
    metric = 'epoch_bleu'
    submetrics_df = metrics_df[metrics_df['Metric'] == metric]
    """ Choose of Language of Interest """
    lang = 'fr'
    submetrics_df = submetrics_df[submetrics_df['Lang'] == lang]
    
    barplot = sns.lineplot(x='Fraction',y='Value',hue='Method',style='Method',lw=3,palette=palette_map,data=submetrics_df,markers=True,markersize=10,markeredgecolor='k',ci=68,dashes=False,ax=axes[1])
    axes[0].remove()
    barplot.set(xlabel='Labelled Fraction')
    barplot.set(ylabel='BLEU')
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.2),ncol=len(metrics_df.Method.value_counts()),markerscale=5)

#%%
def obtain_spec_augment_hyperparams(band_types):
    """ Hyperparameters to Loop Over for Spec Augment Perturbation Experiments """
    if len(band_types) == 1:
        spec_augment_nbands = [[1]]#,[2]]
        spec_augment_band_widths = [[0.2]]#,[0.2]]
    elif len(band_types) > 1:
        spec_augment_nbands = [[1,1]]#,[2,2]] #[1,2],[2,1],[2,2]]
        spec_augment_band_widths = [[0.2,0.2]]#,[0.2,0.2]] #[0.1,0.2],[0.2,0.1],[0.2,0.2]]
    return spec_augment_nbands, spec_augment_band_widths

#%%
basepath_to_data = '/mnt/SecondaryHDD'
phases = ['val']#,'val']#['test'] #['train','val'] #['test']
trial_to_load_list = ['Language_Detection'] #['MLM','ELECTRA','MARGE','Language_Detection']#,'ELECTRA','Language_Detection','Language_Change_Detection'] #['BYOL']#,'Random'] #['SimCLR','CMSC','CMLC','CMSMLC'] #for loading pretrained weights #['Random']
trial_to_run_list =  ['Random','Random','Random','Random'] #['Fine-Tuning']#,'Fine-Tuning'] #['Fine-Tuning','Fine-Tuning','Fine-Tuning','Fine-Tuning'] #['Linear','Linear','Linear','Linear'] #['Fine-Tuning','Fine-Tuning','Fine-Tuning','Fine-Tuning'] #['Linear','Linear','Linear','Linear'] #['Fine-Tuning','Fine-Tuning','Fine-Tuning','Fine-Tuning'] #['Fine-Tuning','Fine-Tuning','Fine-Tuning','Fine-Tuning']  #['Linear','Linear','Linear','Linear']  #['Random']#,'Fine-Tuning','Fine-Tuning','Fine-Tuning']#['SimCLR','CMSC','CMLC','CMSMLC'] #current trial to run and perform training # Fine-Tuning | Same as trial_to_load
embedding_dim_list = [300] #[256,128,64,32]
downstream_dataset_list = ['ptbxl']#['chapman'] #dataset for pretraininng # 'ptbxl' | 'brazil' | 'mimic'
second_dataset_list = ['ptbxl'] #['cardiology','physionet2017','physionet2020'] #only used for fine-tuning & linear trials #keep as list of empty strings if pretraining
labelled_fraction_list = [0.05]#[0.25,0.50,0.75,1.00] #proportion of labelled training data to train on #SHOULD BE 1 for pretraining #[0.25,0.50,0.75,1.00] for finetuning and linear evaluation
downstream_input_perturbed_list = [False]#[True]#[True]#,False]
downstream_perturbation_entries = ['Gaussian','SpecAugment','FlipAlongY','FlipAlongX']
downstream_perturbation_list = list(map(lambda x:[x],downstream_perturbation_entries)) + list(itertools.permutations(downstream_perturbation_entries,2)) #[['Gaussian']] #list of lists containing the following: 'Gaussian' | 'FlipAlongY' | 'FlipAlongX' | 'SpecAugment'
#perturbation_list = perturbation_list[1:]
spec_augment_band_types = [['frequency'],['time'],['frequency','time'],['time','frequency']] #[['frequency'],['time'],['frequency','time'],['time','frequency']] #options: 'frequency' | 'time'
BYOL_tau_list = [0.5] #[0.5,0.9,0.99] #should be of length one for any other experiment to avoid performing redundant loops
""" Perturbations to Apply to Linear or Fine-Tuning Experiments """
second_input_perturbed = False #default
second_perturbation = None #default

#downstream_perturbation_list = downstream_perturbation_list[:4] #single perturbations
#downstream_perturbation_list = downstream_perturbation_list[4:7] #sequential perturbations
#%%
encoder_type = 'cnn_small' #options: 'cnn_small' (default) | 'resnet34' (this can be changed with some modifications)
attn = 'direct_dot_product' #options: 'direct_dot_product' | 'concat_nonlinear' | None (no attention)
auxiliary_loss = False #options: True | False  #(auxiliary class prediction loss for the encoder)
attn_loss = False #options: True | False #(attn loss as in Show Attend and Tell Paper which is regularization term on attention coefs)

goal_list = ['IC']
decoder_type_list = ['transformer']
#dest_lang_list = ['de','el','en','es','fr','it','pt']#,'zh-CN'] #options: 'de' | 'en' | 'fr' | 'el' | 'pt' | 'ja' | 'zh-CN' | 'es' | 'it' (only applicable to image captioning case)
dest_lang_nested_list = [['de','el','en','es','fr','it','pt']] #[['de'],['el'],['en']]#,['el'],['en'],['es'],['fr'],['it'],['pt']]
#decoder_type_list = ['lstm','transformer'] #options: 'lstm' | 'transformer'
num_layers_list = [4] #options: (int) 1 | 2 | etc. layers for the lstm network 
#goal_list = ['Language_Detection','Language_Change_Detection','Text_Supervised'] #options: 'Language_Detection' | 'Language_Change_Detection' | 'Text_Supervised' | 'Supervised' | 'IC' | 'VQA' 
target_token_selection_list = ['uniform']#,'categorical'] #options: 'uniform' | 'categorical' (this determines how to choose tokens from target language when replacing source tokens during pre-training)
replacement_prob_list = [0.15] # fraction of sequence tokens to replace with target token from another language

""" Control These for Pre-Training and Fine-Tuning Purposes """
pretrained_encoder = False #True 
freeze_encoder = False #True

pretrained_decoder = True #True
freeze_decoder = False #False

pretrained_embeddings = True #True
freeze_embeddings = False #False

#%%

""" ****** A ****** Obtain Average Metric Performance Across Seeds """
obtain_metrics_df = True #options: True | False

""" ****** B ****** Obtain Distribution of Target and Output Sentence Tokens """
plot_token_distribution = False #options: True | False

""" ****** C ****** Project Embeddings and Plot Them As Scatter Plot """                                                    
plot_embeddings = False #options: True | False
approach = 'tsne' #options: 'pca' | 'tsne'
k = 10

""" ****** D ****** Obtain Sample Sentences (Target and Predicted) From Each Language """                                                    
obtain_sample_sentences = False #options: True | False
top = False #options: True | False #based on BLEU score (best performing)
bottom = True #options: True | False #based on BLEU score (worst performing)
k = 1

""" ****** E ****** Project Self-supervised Embeddings and Plot Them As Scatter Plot """ 
plot_pretraining_embeddings = False

""" ****** F ****** Perform Cross-Lingual Token Retrieval Using Embedding Dict """                                                    
retrieve_cross_lingual_tokens = False

""" ****** G ****** Perform Token-Based Signal Retrieval and Signal-Based Token Retrieval """
retrieve_signal_and_tokens = False
query_token = 'ischemic' #normal
query_lang = 'en'
k = 10

""" ****** H ****** Plot Signal and Corresponding Multi-lingual Sentences """
plot_signal_and_sentences = False

""" ****** I ****** Plot (Bar) Diversity of Generated Text From Different Methods """
plot_diversity = False

""" ****** J ****** Plot (Line) Effect of Labelled Fraction on Multiple Methods """
plot_more_with_less = False

""" ****** K ****** Plot (Line) Learning Curves of Multiple Methods """
plot_learning_curves = False

#%%

if __name__ == '__main__':
    for input_perturbed in downstream_input_perturbed_list:
        if input_perturbed == False:
            """ No Perturbations Path """
            perturbation = None #filler ['']
            second_input_perturbed = second_input_perturbed
            second_perturbation = second_perturbation 
            """ Results May Need to Be Unpacked Depending on Chosen Evaluation Process """
            results = run_configurations(basepath_to_data,phases,input_perturbed,perturbation,second_input_perturbed,second_perturbation,trial_to_load_list,trial_to_run_list,BYOL_tau_list,embedding_dim_list,downstream_dataset_list,second_dataset_list,labelled_fraction_list,output_type=output_type,input_type=input_type,encoder_type=encoder_type,decoder_type_list=decoder_type_list,
                               num_layers_list=num_layers_list,attn=attn,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,goal_list=goal_list,pretrained_embeddings=pretrained_embeddings,freeze_embeddings=freeze_embeddings,pretrained_encoder=pretrained_encoder,freeze_encoder=freeze_encoder,pretrained_decoder=pretrained_decoder,freeze_decoder=freeze_decoder,dest_lang_nested_list=dest_lang_nested_list,
                               target_token_selection_list=target_token_selection_list,replacement_prob_list=replacement_prob_list,
                               obtain_metrics_df=obtain_metrics_df,plot_token_distribution=plot_token_distribution,
                               plot_embeddings=plot_embeddings,obtain_sample_sentences=obtain_sample_sentences,
                               top=top,bottom=bottom,k=k,approach=approach,retrieve_signal_and_tokens=retrieve_signal_and_tokens,
                               retrieve_cross_lingual_tokens=retrieve_cross_lingual_tokens,
                               query_token=query_token,query_lang=query_lang,
                               plot_pretraining_embeddings=plot_pretraining_embeddings,
                               plot_signal_and_sentences=plot_signal_and_sentences,
                               plot_diversity=plot_diversity,
                               plot_more_with_less=plot_more_with_less,
                               plot_learning_curves=plot_learning_curves)


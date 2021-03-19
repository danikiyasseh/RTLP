#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:40:29 2020

@author: Dani Kiyasseh
"""

#%%
import numpy as np
from prepare_miscellaneous import obtain_information, obtain_saved_weights_name, make_saving_directory, modify_dataset_order_for_multi_task_learning, obtain_load_path_dir, determine_classification_setting
from run_experiment import train_model
import itertools
import argparse
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

#trials_to_load_dict = {
#                'CMC':
#                {'input_perturbed':True, #default is perturbed - do not change
#                 'perturbation':['Gaussian']}, #needs to be a list to allow for sequence of perturbations
#                'SimCLR':
#                {'input_perturbed':True, #default is perturbed - do not change
#                 'perturbation':['Gaussian']}, 
#                'CMSC':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']}, 
#                'CMLC':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']},
#                'CMSMLC':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']},
#                'Linear':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']}, 
#                'Fine-Tuning':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']}, 
#                'Random':
#                {'input_perturbed':False, #default is NO perturbation
#                 'perturbation':['']}, 
#                }
                
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
                       dest_lang_nested_list=[['en']],target_token_selection_list=['uniform'],replacement_prob_list=[0.15]):
    """ Run All Experiments 
    Args:
        phases (list): list of phases for training        
        trial_to_load_list (list): list of trials to load #this is needed for fine-tuning later on
        trial_to_run_list (list): list of trials to run
        embedding_dim_list (list): size of embedding for representation
        downstream_dataset_list (list): list of datasets to perform experiments on
    """
    #band_types=band_types,nbands=nbands,band_width=band_width,
    
    for dest_lang_list in dest_lang_nested_list:
        for trial_to_load,trial_to_run in zip(trial_to_load_list,trial_to_run_list):
            for BYOL_tau in BYOL_tau_list: #will consist of one entry if performing other experiments
                for embedding_dim in embedding_dim_list: #embedding dimension to use for pretraining
                    for downstream_dataset in downstream_dataset_list: #dataset used for pretraining
                        for second_dataset in second_dataset_list: #dataset used for evaluation down the line
                            for labelled_fraction in labelled_fraction_list:
                                
                                """ Iterate Over Decoder Types (LSTM or Transformer) """
                                for decoder_type in decoder_type_list:
                                    """ Iterate Over Number of Layers in Decoder (LSTM ot Transformer) """
                                    for num_layers in num_layers_list:
                                        """ Iterate Over Pre-training Tasks """
                                        for goal in goal_list:
                                            
                                            if trial_to_load in ['Language_Detection','Language_Change_Detection','MLM','ELECTRA']:
                                                replacement_prob_list = replacement_prob_list
                                                target_token_selection_list = target_token_selection_list
                                            else:
                                                replacement_prob_list = ['NA']
                                                target_token_selection_list = ['NA']
                                            
                                            """ Iterate Over Token Replacement Probability """
                                            for replacement_prob in replacement_prob_list:
                                                """ Iterate Over Target Token Selection Strategy """
                                                for target_token_selection in target_token_selection_list:
    
                                                    """ ******* Activate These For Linear OR Fine-Tuning Purposes ******* """
    
                                                    """ Encoder Controls """
                                                    current_pretrained_encoder = {'dataset': downstream_dataset, 'supervision': 'Supervised'} if pretrained_encoder == True else False #options: False | Dict (loads pretrained parameters)
                                                    current_freeze_encoder = freeze_encoder #only applies when pretrained_encoder is not False 
                                                    """ Decoder Controls """
                                                    current_pretrained_decoder = {'dataset': downstream_dataset, 'supervision': trial_to_load, 'langs': dest_lang_list, 'target_token_selection': target_token_selection, 'replacement_prob': replacement_prob, 'num_layers': num_layers} if pretrained_decoder == True else False #options: False | Dict (loads pretrained parameters)
                                                    current_freeze_decoder = freeze_decoder
                                                    """ Word Embeddings Controls """
                                                    current_pretrained_embeddings = {'dataset': downstream_dataset, 'supervision': trial_to_load, 'langs': dest_lang_list, 'target_token_selection': target_token_selection, 'replacement_prob': replacement_prob, 'num_layers': num_layers} if pretrained_embeddings == True else False #options: True | False (should you load pre-trained embeddings - currently this is only based on Spacy embeddings)
                                                    current_freeze_embeddings = freeze_embeddings
    
                                                    downstream_task, nencoders, nviews = trials_to_run_dict[trial_to_run].values()
                                                    #input_perturbed, perturbation = trials_to_load_dict[trial_to_load].values()
                                                    saved_weights = obtain_saved_weights_name(trial_to_run,trial_to_load,phases)
                                                    
                                                    """ Information for save_path_dir """
                                                    original_leads, original_batch_size, original_held_out_lr, original_class_pair, original_modalities, original_fraction = obtain_information(trial_to_load,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict,extra_trial=trial_to_run) #should be trial_to_load alone
                                                    """ Information for actual training --- trial_to_run == trial_to_load when pretraining so they are the same """
                                                    leads, batch_size, held_out_lr, class_pair, modalities, fraction = obtain_information(trial_to_run,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict) #should be trial_to_run alone
                                                    
                                                    max_epochs = 10 #10 for monolingual fine-tuning experiments #25 for multilingual fine-tuning experiments #hard stop for training
                                                    max_seed = 5
                                                    seeds = np.arange(0,max_seed)#max_seed)
                                                    for seed in seeds:
                                                        save_path_dir, seed = make_saving_directory(phases,downstream_dataset,trial_to_load,trial_to_run,seed,max_seed,downstream_task,BYOL_tau,embedding_dim,original_leads,input_perturbed,perturbation,band_types,nbands,band_width,encoder_type,decoder_type,attn,attn_loss,goal,
                                                                                                    current_pretrained_embeddings,current_freeze_embeddings,current_pretrained_encoder,current_freeze_encoder,
                                                                                                    current_pretrained_decoder,current_freeze_decoder,dest_lang_list,target_token_selection,
                                                                                                    replacement_prob,num_layers)
                                                        #if save_path_dir == 'do not train':
                                                        #    continue
                            
                                                        if trial_to_run in ['Linear','Fine-Tuning','Random']:  
                                                            original_downstream_dataset,modalities,leads,class_pair,fraction = modify_dataset_order_for_multi_task_learning(second_dataset,modalities,leads,class_pair,fraction)
                                                        else:
                                                            original_downstream_dataset = downstream_dataset #to avoid overwriting downstream_dataset which is needed for next iterations
                                                        
                                                        load_path_dir, save_path_dir = obtain_load_path_dir(phases,save_path_dir,trial_to_load,trial_to_run,second_input_perturbed,second_perturbation,second_dataset,labelled_fraction,leads,max_seed,downstream_task)
                                                        if save_path_dir in ['do not train','do not test']:
                                                            continue
                                                        
                                                        classification = determine_classification_setting(second_dataset,trial_to_run,output_type)
                                                        train_model(basepath_to_data,classification,load_path_dir,save_path_dir,seed,batch_size,held_out_lr,fraction,modalities,leads,saved_weights,phases,original_downstream_dataset,downstream_task,class_pair,input_perturbed,perturbation,band_types=band_types,
                                                                    nbands=nbands,band_width=band_width,encoder_type=encoder_type,decoder_type=decoder_type,num_layers=num_layers,attn=attn,trial_to_load=trial_to_load,trial_to_run=trial_to_run,nencoders=nencoders,BYOL_tau=BYOL_tau,embedding_dim=embedding_dim,nviews=nviews,labelled_fraction=labelled_fraction,
                                                                    output_type=output_type,input_type=input_type,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,goal=goal,pretrained_embeddings=current_pretrained_embeddings,freeze_embeddings=current_freeze_embeddings,pretrained_encoder=current_pretrained_encoder,freeze_encoder=current_freeze_encoder,
                                                                    pretrained_decoder=current_pretrained_decoder,freeze_decoder=current_freeze_decoder,dest_lang_list=dest_lang_list,target_token_selection=target_token_selection,replacement_prob=replacement_prob,num_epochs=max_epochs)

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
phases = ['train','val']#,'val']#['test'] #['train','val'] #['test']
trial_to_load_list = ['Language_Detection'] #['Language_Detection']  #'MARGE','MLM','ELECTRA']#,#['BYOL']#,'Random'] #['SimCLR','CMSC','CMLC','CMSMLC'] #for loading pretrained weights #['Random']
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

parser = argparse.ArgumentParser()
parser.add_argument('-goal_list',nargs='+',help='input pre-training task')
parser.add_argument('-decoder_type_list',nargs='+',help='input type of decoder to use')
args = parser.parse_args()
goal_list = args.goal_list
decoder_type_list = args.decoder_type_list

dest_lang_nested_list = [['de','el','en','es','fr','it','pt']] #[['el'],['en'],['es'],['fr'],['it'],['pt']] #[['de','el','en','es','fr','it','pt']] #,'zh-CN'] #options: 'de' | 'en' | 'fr' | 'el' | 'pt' | 'ja' | 'zh-CN' | 'es' | 'it' (only applicable to image captioning case)
#decoder_type_list = ['lstm','transformer'] #options: 'lstm' | 'transformer'
num_layers_list = [4] #options: (int) 1 | 2 | etc. layers for the lstm network 
#goal_list = ['Language_Detection','Language_Change_Detection','Text_Supervised'] #options: 'Language_Detection' | 'Language_Change_Detection' | 'Text_Supervised' | 'Supervised' | 'IC' | 'VQA' 
target_token_selection_list = ['uniform']#,'categorical'] #options: 'uniform' | 'categorical' (this determines how to choose tokens from target language when replacing source tokens during pre-training)
replacement_prob_list = [0.15] # fraction of sequence tokens to replace with target token from another language

""" Control These for Pre-Training and Fine-Tuning Purposes """
pretrained_encoder = True #True for downstream 
freeze_encoder = False #True for downstream

pretrained_decoder = True #True for downstream
freeze_decoder = False #False for downstream

pretrained_embeddings = True #True for downstream
freeze_embeddings = False #False for downstream

if __name__ == '__main__':
    for input_perturbed in downstream_input_perturbed_list:
#        if input_perturbed == True:
#            """ Iterate Over All Perturbations """
#            for perturbation in downstream_perturbation_list: #note, perturbation can also be a list
#                
#                """ Determine Actual Perturbation Applied to Inputs """
#                if trial_to_load_list == trial_to_run_list: #implies we are in the pre-training phase
#                    second_input_perturbed = input_perturbed #use downstream... for saving directory #use second for actual training
#                    second_perturbation = perturbation
#                else: #perturbation for linear or fine-tuning experiments 
#                    second_input_perturbed = second_input_perturbed
#                    second_perturbation = second_perturbation 
#                
#                """ SpecAugment-related Experiments """
#                if 'SpecAugment' in perturbation:
#                    """ SpecAugment Perturbations """
#                    for band_types in spec_augment_band_types:
#                        spec_augment_nbands, spec_augment_band_widths = obtain_spec_augment_hyperparams(band_types)
#                        for nbands in spec_augment_nbands:
#                            for band_width in spec_augment_band_widths:
#                                run_configurations(basepath_to_data,phases,input_perturbed,perturbation,second_input_perturbed,second_perturbation,trial_to_load_list,trial_to_run_list,BYOL_tau_list,embedding_dim_list,downstream_dataset_list,second_dataset_list,labelled_fraction_list,band_types=band_types,
#                                                   nbands=nbands,band_width=band_width,output_type=output_type,input_type=input_type,encoder_type=encoder_type,decoder_type_list=decoder_type_list,num_layers_list=num_layers_list,attn=attn,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,goal_list=goal_list,pretrained_embeddings=pretrained_embeddings,
#                                                   freeze_embeddings=freeze_embeddings,pretrained_encoder=pretrained_encoder,freeze_encoder=freeze_encoder,pretrained_decoder=pretrained_decoder,freeze_decoder=freeze_decoder,dest_lang_list=dest_lang_list,target_token_selection_list=target_token_selection_list,replacement_prob_list=replacement_prob_list)
#                else:
#                    """ Perturbations OTHER Than SpecAugment """
#                    run_configurations(basepath_to_data,phases,input_perturbed,perturbation,second_input_perturbed,second_perturbation,trial_to_load_list,trial_to_run_list,BYOL_tau_list,embedding_dim_list,downstream_dataset_list,second_dataset_list,labelled_fraction_list,output_type=output_type,input_type=input_type,encoder_type=encoder_type,decoder_type_list=decoder_type_list,
#                                       num_layers_list=num_layers_list,attn=attn,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,goal_list=goal_list,pretrained_embeddings=pretrained_embeddings,freeze_embeddings=freeze_embeddings,pretrained_encoder=pretrained_encoder,freeze_encoder=freeze_encoder,pretrained_decoder=pretrained_decoder,freeze_decoder=freeze_decoder,dest_lang_list=dest_lang_list,
#                                       target_token_selection_list=target_token_selection_list,replacement_prob_list=replacement_prob_list)                    

        if input_perturbed == False:
            """ No Perturbations Path """
            perturbation = None #filler ['']
            second_input_perturbed = second_input_perturbed
            second_perturbation = second_perturbation 
            run_configurations(basepath_to_data,phases,input_perturbed,perturbation,second_input_perturbed,second_perturbation,trial_to_load_list,trial_to_run_list,BYOL_tau_list,embedding_dim_list,downstream_dataset_list,second_dataset_list,labelled_fraction_list,output_type=output_type,input_type=input_type,encoder_type=encoder_type,decoder_type_list=decoder_type_list,
                               num_layers_list=num_layers_list,attn=attn,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,goal_list=goal_list,pretrained_embeddings=pretrained_embeddings,freeze_embeddings=freeze_embeddings,pretrained_encoder=pretrained_encoder,freeze_encoder=freeze_encoder,pretrained_decoder=pretrained_decoder,freeze_decoder=freeze_decoder,dest_lang_nested_list=dest_lang_nested_list,
                               target_token_selection_list=target_token_selection_list,replacement_prob_list=replacement_prob_list)




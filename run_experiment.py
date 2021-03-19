#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:28:57 2020

@author: Dani Kiyasseh
"""
import torch
import os
import numpy as np
import copy

from prepare_miscellaneous import obtain_criterion, print_metrics, track_metrics, \
                                    save_metrics, save_config_weights, save_patient_representation, \
                                    save_predicted_sentences, save_target_sentences
from prepare_models import load_initial_model
from prepare_dataloaders import Load_Data
from perform_training import one_epoch_finetuning
#%%
""" Functions in this script:
    1) train_model
"""
#%%

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad_(False)

def train_model(basepath_to_data,classification,load_path_dir,save_path_dir,seed,batch_size,held_out_lr,fraction,modalities,leads,saved_weights,phases,downstream_dataset,downstream_task,class_pair,second_input_perturbed,second_perturbation,band_types=['frequency','time'],nbands=[1,1],band_width=[0.1,0.1],
                encoder_type='cnn_small',decoder_type='lstm',num_layers=1,attn=False,trial_to_load=None,trial_to_run=None,nencoders=1,BYOL_tau=0.99,embedding_dim=256,nviews=1,labelled_fraction=1,output_type='single',input_type='single',auxiliary_loss=False,attn_loss=False,
                goal='IC',pretrained_embeddings=False,freeze_embeddings=False,pretrained_encoder=False,freeze_encoder=False,pretrained_decoder=False,freeze_decoder=False,dest_lang_list=['en'],target_token_selection='uniform',replacement_prob=0.15,num_epochs=250):
    
    """ Training and Validation For All Epochs """
    if goal in ['IC','VQA']:
        best_metric = 0
    elif goal in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
        best_metric = float('inf')
        
    metrics_dict = dict()
    #patient_rep_dict = dict()
    if 'train' in phases:
        if len(phases) == 1:
            phases = ['train1']
            inferences = [False]
        else:
            phases = ['train1','val']
            inferences = [False,False]            
    elif 'val' in phases and len(phases) == 1:
        phases = ['val']
        inferences = [False]        
    else:
        inferences = [False]
    
    stop_counter = 0
    patience = 10 #for early stopping criterion
    epoch_count = 0
    
    #if downstream_dataset == 'ptbxl':
    #    patience = 15
    
    criterion = obtain_criterion(classification)
    if 'train1' in phases or 'val' in phases:
        model_path_dir = load_path_dir #original use-case
    elif 'test' in phases:
        model_path_dir = save_path_dir
    
    """ Load Model and Word Embeddings """
    model, token2id_dict, id2embedding_dict, optimizer, device, df = load_initial_model(phases,trial_to_load,model_path_dir,saved_weights,held_out_lr,nencoders,embedding_dim,downstream_dataset,encoder_type=encoder_type,decoder_type=decoder_type,num_layers=num_layers,attn=attn,input_type=input_type,
                                                                                    goal=goal,pretrained_embeddings=pretrained_embeddings,freeze_embeddings=freeze_embeddings,pretrained_encoder=pretrained_encoder,freeze_encoder=freeze_encoder,pretrained_decoder=pretrained_decoder,
                                                                                    freeze_decoder=freeze_decoder,classification=classification,dest_lang_list=dest_lang_list,target_token_selection=target_token_selection,replacement_prob=replacement_prob,labelled_fraction=labelled_fraction)
        
    weighted_sampling = []
    acquired_indices = [] #indices of the unlabelled data
    acquired_labels = dict() #network labels of the unlabelled data
    
    """ Load Dataset """
    print('Loading Dataset...')
    load_data = Load_Data(downstream_dataset,goal=goal)
    dataloader, operations = load_data.load_initial_data(basepath_to_data,phases,df,fraction,inferences,batch_size,modalities,acquired_indices,acquired_labels,modalities,token2id_dict,downstream_task=downstream_task,input_perturbed=second_input_perturbed,perturbation=second_perturbation,
                                               band_types=band_types,nbands=nbands,band_width=band_width,leads=leads,class_pair=class_pair,trial=trial_to_run,nviews=nviews,labelled_fraction=labelled_fraction,output_type=output_type,input_type=input_type,dest_lang_list=dest_lang_list)

    """ Obtain Number of Labelled Samples """
    #total_labelled_samples = len(dataloaders_list['train1'].batch_sampler.sampler.data_source.label_array)
    
    while stop_counter <= patience and epoch_count < num_epochs:
        if 'train' in phases or 'val' in phases:
            print('-' * 10)
            print('Epoch %i/%i' % (epoch_count,num_epochs-1))
            print('-' * 10)
                            
        """ ACTUAL TRAINING AND EVALUATION """
        for phase,inference in zip(phases,inferences):
            if 'train1' in phase:
                model.train()
            elif phase == 'val' or phase == 'test':
                model.eval()
            #labels is the ground-truth text.  # modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
            if goal in ['VQA','Supervised','Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
                results_dictionary = one_epoch_finetuning(weighted_sampling,phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,criterion,classification,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,trial=trial_to_run,epoch_count=epoch_count,goal=goal,save_path_dir=save_path_dir)
            elif goal in ['IC']:
                results_dictionary, outputs_list, labels_list = one_epoch_finetuning(weighted_sampling,phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,criterion,classification,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,trial=trial_to_run,epoch_count=epoch_count,goal=goal,save_path_dir=save_path_dir)
                
            
            if inference == False:
                print_metrics(phase,results_dictionary)
                #epoch_loss = results_dictionary['epoch_loss']
                
                if goal in ['IC','VQA']:
                    epoch_metric_name = 'epoch_bleu' if goal == 'IC' else 'epoch_acc'
                    #print(list(results_dictionary[epoch_metric_name].values()))
                    epoch_metric = np.mean(list(results_dictionary[epoch_metric_name].values()))
                    """ Early Stopping Criterion for IC or VQA """
                    early_stopping_condition_A = epoch_metric >= best_metric #I am more lenient in this case
                    early_stopping_condition_B = epoch_metric < best_metric
                elif goal in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
                    epoch_metric_name = 'epoch_loss'
                    epoch_metric = results_dictionary[epoch_metric_name] #only a scalar because loss is across languages 
                    """ Early Stopping Criterion for Supervised Training """
                    early_stopping_condition_A = epoch_metric < best_metric
                    early_stopping_condition_B = epoch_metric >= best_metric
                
                if (phase == 'val' and early_stopping_condition_A) or (phase == 'test' and early_stopping_condition_A):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_id2embedding_dict = id2embedding_dict
                    if goal in ['IC']:
                        best_outputs_list = outputs_list
                        best_labels_list = labels_list
#                    best_representations_list = representations_list
#                    if goal not in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection']:
#                        #best_attn_coefs_list = np.vstack(attn_coefs_list)
#                        best_attn_coefs_list = attn_coefs_list
                        
                    """ Save Best Finetuned Weights """
                    if 'train1' in phases: #only save weights when training is part of the process
                        save_config_weights(save_path_dir,best_model_wts,saved_weights,phases,trial_to_run,downstream_dataset)
                    stop_counter = 0
                elif phase == 'val' and early_stopping_condition_B:
                    stop_counter += 1
                
                metrics_dict = track_metrics(metrics_dict,results_dictionary,phase,epoch_count)                

        epoch_count += 1
        if 'train1' not in phases:
            break #from while loop
        
        #if epoch_count == 2:
        #    break

    if 'train1' in phases or 'val' in phases:
        if 'train1' in phases:
            prefix = 'train_val'
        elif 'val' in phases:
            prefix = 'val'

        """ Save Regardless of Goal """
#        torch.save(best_representations_list,os.path.join(save_path_dir,'representations'))
        save_metrics(save_path_dir,prefix,metrics_dict)
        if goal in ['IC']:
            torch.save(best_outputs_list,os.path.join(save_path_dir,'output_sentences'))
            torch.save(best_labels_list,os.path.join(save_path_dir,'target_sentences'))
        
        if goal not in ['Supervised']:
            torch.save(best_id2embedding_dict,os.path.join(save_path_dir,'id2embedding_dict'))                
#            if goal not in ['Text_Supervised','Language_Change_Detection','Language_Detection']:
#                torch.save(best_attn_coefs_list,os.path.join(save_path_dir,'attn_coefs')) 
                
        model.load_state_dict(best_model_wts)
    elif 'test' in phases:
        prefix = 'test'
        save_metrics(save_path_dir,prefix,metrics_dict)
        
        
        
        
        
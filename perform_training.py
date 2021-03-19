#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:24:46 2020

@author: Dani Kiyasseh
"""
#import numpy as np
#import os
import random
import torch.nn as nn
import torch
from tqdm import tqdm
from prepare_miscellaneous import obtain_contrastive_loss, flatten_arrays, \
                                    calculate_auc, change_labels_type, obtain_auto_encoder_loss, \
                                    obtain_BYOL_loss, calculate_acc, calculate_bleu_score, \
                                    calculate_answer_accuracy, calculate_rouge_score, \
                                    calculate_meteor_score, calculate_IC_loss, calculate_language_detection_accuracy, \
                                    convert_predicted_ids_to_sentences, convert_target_ids_to_sentences

#%%
""" Functions in this script:
    3) one_epoch_finetuning
    4) finetuning_single
"""
#%%

def finetuning_single(phase,token2id_dict,id2embedding_dict,inference,dataloaders,model,optimizer,device,weighted_sampling,criterion,classification,auxiliary_loss=False,attn_loss=False,epoch_count=None,new_task_epochs=None,trial=None,goal='IC',save_path_dir=None): #b/c it is single, models_list contains one model only
    """ One Epoch's Worth of Training for Contrastive Learning Paradigm """
    running_loss = 0.0
    
#    outputs_list = []
#    representations_list = []
#    labels_list = []
#    modality_list = []
#    indices_list = []
#    task_names_list = []
#    attn_coefs_list = []
#    sentence_lens_list = []
#    class_labels_list = []
#    class_predictions_list = []
    
    """ Initialize Dictionaries to Store Results """    
    outputs_dict = dict()
    representations_dict = dict()
    attn_coefs_dict = dict()
    labels_dict = dict()
    sentence_lens_dict = dict()
    class_labels_dict = dict()
    class_predictions_dict = dict()
    epoch_bleu = dict()
    epoch_rouge = dict()
    epoch_meteor = dict()

    for dest_lang in token2id_dict.keys():
        outputs_dict[dest_lang] = list()
        attn_coefs_dict[dest_lang] = list()
        representations_dict[dest_lang] = list()
        labels_dict[dest_lang] = list()
        sentence_lens_dict[dest_lang] = list()
        class_labels_dict[dest_lang] = list()
        class_predictions_dict[dest_lang] = list()
        epoch_bleu[dest_lang] = 0
        epoch_rouge[dest_lang] = 0
        epoch_meteor[dest_lang] = 0

    batch_num = 0
    batch = 0
    #class label is that in IC setting, but class label is answer in VQA setting
    for inputs, text_indices, sentence_lens, class_labels, languages, document_level_text_indices, document_level_sentence_lens in tqdm(dataloaders[phase]):
        """ Weaning Off of Teacher Forcing in a Linear Manner """
        #sampling_prob = (0.4/30000)*(batch+1)*(epoch_count+1)
        #uniform_value = np.random.uniform(0,1)
        #sampling = True if uniform_value < sampling_prob else False
        sampling = False
        batch += 1
        """ Send Data to Device """
        inputs = inputs.to(device)
        class_labels = class_labels.to(device)
        #print(text_indices)
        with torch.set_grad_enabled('train1' in phase):# and inference == False): #('train' in phase and inference == False)
            """ Image Captioning Path """
            if goal == 'IC':
                """ Perform Forward Pass i.e. Encoder and Decoder """
                current_labels_dict = dict() #text
#                current_class_labels_dict = dict()
#                current_class_predictions_dict = dict()
                current_outputs_dict = dict()
#                current_attn_coefs_dict = dict()
#                current_representations_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()): #, sorted_indices, attn_coefs, class_predictions
                    outputs, representations = model(inputs,current_text_indices,current_sentence_lens,token2id_dict[dest_lang],id2embedding_dict[dest_lang],dest_lang,phase,sampling,device) #outputs is B x S x Words
                    """ Convert Text Indices/Targets to Tensor """
                    current_text_indices = current_text_indices.to(device) #torch.tensor(current_text_indices,device=device)
                    """ Remove '/START' Index from Target Indices """
                    #current_text_indices = current_text_indices[:,1:] # B x (S-1)
                    if phase == 'train1':
                        attn_coefs = 5
                        class_predictions = 6
                        loss = calculate_IC_loss(criterion,outputs,current_text_indices[:,1:],class_predictions,class_labels,attn_coefs,auxiliary_loss,attn_loss)
                        total_loss = total_loss + loss
                        """ Average Loss if This is Final Loss Collected """
                        if dest_lang == list(text_indices.keys())[-1]:
                            loss = total_loss / len(text_indices)
                    """ Store Results """
                    current_labels_dict[dest_lang] = current_text_indices[:,1:].cpu().detach().numpy()
#                    current_class_labels_dict[dest_lang] = class_labels
#                    current_class_predictions_dict[dest_lang] = class_predictions
                    current_outputs_dict[dest_lang] = outputs.cpu().detach().numpy() #text
#                    current_attn_coefs_dict[dest_lang] = attn_coefs
#                    current_representations_dict[dest_lang] = representations
                    #""" Detach Outputs and Attn Coefs To Avoid Memory Leakage """
                    #outputs = outputs.detach()
                    #attn_coefs = attn_coefs.detach()
                current_text_indices.detach()
            elif goal == 'VQA':
                """ Perform Forward Pass and Get Answers """
                outputs, representations, attn_coefs, class_predictions = model(inputs,text_indices,sentence_lens,id2embedding_dict,phase,device)
                """ Calculate MSE Loss """
                #criterion = nn.MSELoss()
                #class_labels = class_labels.type(torch.float)
                """ Calculate CrossEntropyLoss """
                criterion = nn.CrossEntropyLoss()
                class_labels = class_labels.type(torch.long)
                #print(outputs,outputs.shape)
                loss = criterion(outputs,class_labels)
            elif goal == 'Supervised': #encoder supervised pre-training
                h, representations, class_predictions = model(inputs)#,text_indices,sentence_lens,id2embedding_dict,phase,device)
                criterion = nn.CrossEntropyLoss()
                class_labels = class_labels.type(torch.long)
                loss = criterion(class_predictions,class_labels)
            elif goal == 'Text_Supervised':
                #h, class_predictions = model.supervised_forward(text_indices,sentence_lens,token2id_dict,id2embedding_dict,phase,device)
                criterion = nn.CrossEntropyLoss()
                class_labels = class_labels.type(torch.long)
                current_class_labels_dict = dict()
                current_class_predictions_dict = dict()
#                current_representations_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()):
                    class_predictions = model.supervised_forward(current_text_indices,current_sentence_lens,token2id_dict[dest_lang],id2embedding_dict[dest_lang],phase,device)
                    loss = criterion(class_predictions,class_labels)
                    total_loss = total_loss + loss
                    """ Average Loss if This is Final Loss Collected """
                    if dest_lang == list(text_indices.keys())[-1]:
                        loss = total_loss / len(text_indices)

                    current_class_labels_dict[dest_lang] = class_labels.cpu().detach().numpy()
                    current_class_predictions_dict[dest_lang] = class_predictions.cpu().detach().numpy()
#                    current_representations_dict[dest_lang] = h
                #loss = criterion(class_predictions,class_labels)
                #print(loss)
            elif goal == 'Language_Change_Detection':
                criterion = nn.BCEWithLogitsLoss()
                class_labels = class_labels.type(torch.long)
                current_class_labels_dict = dict()
                current_class_predictions_dict = dict()
#                current_representations_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()):
                    """ Forward Pass """
                    replacement_predictions, replacement_labels = model.language_change_detection_forward(current_text_indices,current_sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device)
                    #replacement_labels = replacement_labels.type(torch.float) #needed for BCELoss
                    """ Instance-Wise Loss Because Each Sentence is of a Different Length """
                    loss = 0
                    for i,(replacement_prediction,replacement_label) in enumerate(zip(replacement_predictions,replacement_labels)):
                        current_loss = criterion(replacement_prediction,replacement_label)
                        loss = loss + current_loss
                        if i == len(replacement_predictions)-1:
                            loss = loss / len(replacement_predictions)
                    #loss = torch.mean(torch.tensor([criterion(replacement_prediction,replacement_label) for replacement_prediction,replacement_label in zip(replacement_predictions,replacement_labels)]))
                    total_loss = total_loss + loss
                    """ Average Loss if This is Final Loss Collected """
                    if dest_lang == list(text_indices.keys())[-1]:
                        loss = total_loss / len(text_indices)
                        
                    """ Store Representations and Labels """
                    current_class_predictions_dict[dest_lang] = [predictions.cpu().detach().numpy() for predictions in replacement_predictions]
                    current_class_labels_dict[dest_lang] = [labels.cpu().detach().numpy() for labels in replacement_labels]
#                    current_representations_dict[dest_lang] = h 
            elif goal == 'Language_Detection':
                criterion = nn.CrossEntropyLoss(ignore_index=0)
                class_labels = class_labels.type(torch.long)
                current_class_labels_dict = dict()
                current_class_predictions_dict = dict()
#                current_representations_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()):
                    """ Forward Pass """
                    replacement_predictions, replacement_labels = model.language_detection_forward(current_text_indices,current_sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device)
                    #replacement_labels = replacement_labels.type(torch.long) #needed for CrossEntropyLoss
                    """ Instance-Wise Loss Because Each Sentence is of a Different Length """
#                    loss = 0
#                    for i,(replacement_prediction,replacement_label) in enumerate(zip(replacement_predictions,replacement_labels)):
#                        replacement_label = replacement_label.type(torch.long)
#                        current_loss = criterion(replacement_prediction,replacement_label)
#                        loss = loss + current_loss
#                        if i == len(replacement_predictions)-1:
#                            loss = loss / len(replacement_predictions)
                    #print(replacement_predictions.shape,replacement_labels.shape)
                    loss = criterion(replacement_predictions.permute(0,2,1),replacement_labels)
                    #print(loss)
                    total_loss = total_loss + loss
                    #print(dest_lang,total_loss)
                    """ Average Loss if This is Final Loss Collected """
                    if dest_lang == list(text_indices.keys())[-1]:
                        loss = total_loss / len(text_indices)
                        
                    """ Store Representations and Labels """
                    current_class_predictions_dict[dest_lang] = [predictions.cpu().detach().numpy() for predictions in replacement_predictions]
                    current_class_labels_dict[dest_lang] = [labels.cpu().detach().numpy() for labels in replacement_labels]
#                    current_representations_dict[dest_lang] = h
            elif goal == 'MLM':
                criterion = nn.CrossEntropyLoss(reduction='none')
#                current_labels_dict = dict() #text
#                current_outputs_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()): #, sorted_indices, attn_coefs, class_predictions
                    outputs, replacement_predictions = model.MLM_forward(current_text_indices,current_sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device) #outputs is B x S x Words
                    """ Convert Text Indices/Targets to Tensor """
                    current_text_indices = current_text_indices.to(device) #torch.tensor(current_text_indices,device=device)
                    """ Remove '/START' Index from Target Indices """
                    current_text_indices = current_text_indices[:,1:] # B x (S-1)
                    """ Obtain Applicable Loss Locations (i.e., Where Token Was Masked) """
                    token_loss_mask = torch.where(replacement_predictions == 1,torch.tensor(1,device=device),torch.tensor(0,device=device)).type(torch.bool)
                    #print(outputs.shape)
                    #if phase == 'train1':
                    """ Obtain Each Token's Loss """
                    token_loss = criterion(outputs.permute(0,2,1),current_text_indices)
                    """ Retrieve Only Relevant Losses (Masked) """
                    loss = torch.mean(token_loss.masked_select(token_loss_mask))
                    """ Aggregate Loss Across Languages """
                    total_loss = total_loss + loss
                    """ Average Loss if This is Final Loss Collected """
                    if dest_lang == list(text_indices.keys())[-1]:
                        loss = total_loss / len(text_indices)
                
                del current_text_indices
                del token_loss
                del token_loss_mask
#                    """ Store Results """
#                    current_labels_dict[dest_lang] = current_text_indices.cpu().detach().numpy()
#                    current_outputs_dict[dest_lang] = outputs.cpu().detach().numpy() #text
            elif goal == 'ELECTRA':
                generator_criterion = nn.CrossEntropyLoss(reduction='none')
                discriminator_criterion = nn.BCEWithLogitsLoss(reduction='none')
#                current_labels_dict = dict() #text
#                current_outputs_dict = dict()
                total_loss = 0
                for (dest_lang,current_text_indices),current_sentence_lens in zip(text_indices.items(),sentence_lens.values()): #, sorted_indices, attn_coefs, class_predictions
                    """ Convert Text Indices/Targets to Tensor """
                    current_text_indices = current_text_indices.to(device) #torch.tensor(current_text_indices,device=device)
                    """ Perform Forward Pass Through ELECTRA """
                    generator_outputs, generator_labels, discriminator_outputs, discriminator_labels = model.ELECTRA_forward(current_text_indices,current_sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device) #outputs is B x S x Words
                    """ Remove '/START' Index from Target Indices """
                    current_text_indices = current_text_indices[:,1:] # B x (S-1)
                    """ Generator Loss Mask (i.e., Only Consider Originally Masked Tokens ) """
                    generator_token_loss_mask = torch.where(generator_labels == 1,torch.tensor(1,device=device),torch.tensor(0,device=device)).type(torch.bool)
                    """ Discrimiantor Loss Mask (i.e., Do Not Consider Padded Regions ) """
                    discriminator_labels = discriminator_labels.view_as(discriminator_outputs)                       
                    discriminator_token_loss_mask = torch.ones_like(discriminator_labels)
                    for i,sentence_len in zip(range(discriminator_token_loss_mask.shape[0]),current_sentence_lens):
                        discriminator_token_loss_mask[i,sentence_len:] = 0
                        
                    #if phase == 'train1':
                    """ Obtain Each Generator Token's Loss """
                    generator_token_loss = generator_criterion(generator_outputs.permute(0,2,1),current_text_indices) # B x S
                    #print(generator_token_loss.shape,generator_token_loss_mask.shape)
                    """ Retrieve Only Relevant Loss (Masked) """
                    generator_loss = torch.mean(generator_token_loss.masked_select(generator_token_loss_mask)) #scalar
                    
                    """ Obtain Each Discriminator Token's Loss """ 
                    discriminator_token_loss = discriminator_criterion(discriminator_outputs,discriminator_labels) # B x S
                    #print(discriminator_token_loss.shape,discriminator_token_loss_mask.shape)
                    """ Retrieve Only Relevant Loss (Masked) """
                    discriminator_loss = torch.mean(discriminator_token_loss.masked_select(discriminator_token_loss_mask.type(torch.bool))) #scalar
                       
                    #print(generator_loss,discriminator_loss)
                    """ Aggregate Loss Across Languages """
                    total_loss = total_loss + generator_loss + discriminator_loss
                    """ Average Loss if This is Final Loss Collected """
                    if dest_lang == list(text_indices.keys())[-1]:
                        loss = total_loss / len(text_indices)
                    """ Store Results """
#                    current_labels_dict[dest_lang] = discriminator_labels.cpu().detach().numpy()
#                    current_outputs_dict[dest_lang] = discriminator_outputs.cpu().detach().numpy() #text
            elif goal == 'MARGE':
#                current_labels_dict = dict() #text
#                current_outputs_dict = dict()
                #total_loss = 0
                #for (dest_lang,current_text_indices),current_sentence_lens,current_languages in zip(text_indices.items(),sentence_lens.values(),languages.values()): #, sorted_indices, attn_coefs, class_predictions
                """ Randomly Choose Target Lang for This Mini-Batch """
                #lang_list = list(text_indices.keys())
                #target_lang = random.sample(lang_list,1).item()
                #target_lang = 'de' #option to change based on dataset (MUST CHANGE IN PAD COLLATE)
                outputs, target_lang = model(text_indices,sentence_lens,languages,document_level_text_indices,document_level_sentence_lens,token2id_dict,id2embedding_dict,phase,device)
                """ Convert Text Indices/Targets to Tensor """
                current_text_indices = text_indices[target_lang].to(device) #torch.tensor(current_text_indices,device=device)
                """ Remove '/START' Index from Target Indices """
                current_text_indices = current_text_indices[:,1:] # B x (S-1)
                #if phase == 'train1':
                """ Obtain Each Token's Loss """
                loss = criterion(outputs.permute(0,2,1),current_text_indices)
                #print(loss)
                    #""" Aggregate Loss Across Languages """
                    #total_loss = total_loss + loss
                    #""" Average Loss if This is Final Loss Collected """
                    #if dest_lang == list(text_indices.keys())[-1]:
                    #    loss = total_loss / len(text_indices)
#                print(loss)
#                """ Store Results """
#                current_labels_dict[target_lang] = current_text_indices.cpu().detach().numpy()
#                current_outputs_dict[target_lang] = outputs.cpu().detach().numpy() #text
            

        """ Backpropagation and Update Step """
        if phase == 'train1': #only perform backprop for train1 phase           
            loss.backward()
            
            """ Network Parameters """
            if isinstance(optimizer,tuple):
                optimizer[0].step()
                """ Task-Instance Parameters """
                optimizer[1].step()                
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        
        """ Calculate Metrics """
        if goal == 'IC':
            if phase == 'train1':
                running_loss += loss.item() * inputs.shape[0]
        elif goal == 'VQA':
            running_loss += loss.item() * inputs.shape[0] 
        elif goal in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
            running_loss += loss.item() * inputs.shape[0]             
        
#        """ These Need to be Language Specific """
            
        if goal in ['IC']:
            batch_bleu = calculate_bleu_score(current_outputs_dict,current_labels_dict,token2id_dict)
            batch_rouge = calculate_rouge_score(current_outputs_dict,current_labels_dict,token2id_dict)
            batch_meteor = calculate_meteor_score(current_outputs_dict,current_labels_dict,token2id_dict) 
            
            for dest_lang in batch_bleu.keys():
                epoch_bleu[dest_lang] = epoch_bleu[dest_lang] + (1/batch)*(batch_bleu[dest_lang] - epoch_bleu[dest_lang])
                epoch_rouge[dest_lang] = epoch_rouge[dest_lang] + (1/batch)*(batch_rouge[dest_lang] - epoch_rouge[dest_lang])
                epoch_meteor[dest_lang] = epoch_meteor[dest_lang] + (1/batch)*(batch_meteor[dest_lang] - epoch_meteor[dest_lang])
            
            if phase in ['val']:
                for dest_lang in text_indices.keys():
                    predicted_sentences = convert_predicted_ids_to_sentences(current_outputs_dict[dest_lang],token2id_dict[dest_lang],dest_lang)
                    target_sentences = convert_target_ids_to_sentences(current_labels_dict[dest_lang],token2id_dict[dest_lang],dest_lang)
                    outputs_dict[dest_lang].extend(predicted_sentences)
                    labels_dict[dest_lang].extend(target_sentences)
            
        elif goal in ['Language_Change_Detection','Language_Detection']:
            for dest_lang in text_indices.keys():
                if goal in ['Language_Change_Detection','Language_Detection']:
                    """ Store Batch Data in The Dictionaries """
                    class_labels_dict[dest_lang].extend(current_class_labels_dict[dest_lang]) #.cpu().detach().numpy())
                    class_predictions_dict[dest_lang].extend(current_class_predictions_dict[dest_lang]) #.cpu().detach().numpy())
               
#            elif goal in ['Text_Supervised']:
##                current_class_labels = current_class_labels_dict[dest_lang]
##                current_class_predictions = current_class_predictions_dict[dest_lang]
##                current_class_labels = current_class_labels.cpu().detach().numpy()
##                current_class_predictions = current_class_predictions.cpu().detach().numpy()
#            
#                """ Store Batch Data in The Dictionaries """
#                #sentence_lens_dict[dest_lang].extend(current_sentence_lens)
#                class_labels_dict[dest_lang].extend(current_class_labels_dict[dest_lang]) #.cpu().detach().numpy())
#                class_predictions_dict[dest_lang].extend(current_class_predictions_dict[dest_lang]) #.cpu().detach().numpy())
#
#            elif goal in ['MARGE']:
#                labels_dict[target_lang].extend(current_labels_dict[target_lang]) #.cpu().detach().numpy())
#                outputs_dict[target_lang].extend(current_outputs_dict[target_lang]) #.cpu().detach().numpy())
#                break # because only one target language per minibatch                
#            if goal not in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection']:
##                if current_labels_dict[dest_lang].data.dtype != torch.long:
##                    current_labels_dict[dest_lang].data = current_labels_dict[dest_lang].data.type(torch.long)
#    
##                current_text_indices = current_labels_dict[dest_lang]
##                current_outputs = current_outputs_dict[dest_lang]
##                current_attn_coefs = current_attn_coefs_dict[dest_lang]
##                current_representations = current_representations_dict[dest_lang]
#                """ Store Batch Data in The Dictionaries """ 
#                labels_dict[dest_lang].extend(current_labels_dict[dest_lang]) #.cpu().detach().numpy())
#                outputs_dict[dest_lang].extend(current_outputs_dict[dest_lang]) #.cpu().detach().numpy())
##                attn_coefs_dict[dest_lang].extend(current_attn_coefs.cpu().detach().numpy())
##                representations_dict[dest_lang].extend(current_representations.cpu().detach().numpy())
##            elif goal in ['Text_Supervised']:
##                current_representations = current_representations_dict[dest_lang]
##                representations_dict[dest_lang].extend(current_representations.squeeze().cpu().detach().numpy())        
##            else:
##                current_representations = current_representations_dict[dest_lang]
##                if goal in ['Language_Change_Detection','Language_Detection']:
##                    current_representations = [representations.cpu().detach().numpy() for representations in current_representations]
##                else:
##                    current_representations = current_representations.cpu().detach().numpy()
##                representations_dict[dest_lang].extend(current_representations)        
#                
##        modality_list.append(modality)
##        indices_list.append(indices)
##        task_names_list.append(task_names)
        
        batch_num += 1
        #if batch_num == 2:
        #    break
        
    #outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list)
    if goal == 'IC':
        if phase == 'train1':
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
        else:
            epoch_loss = 0 #filler
    elif goal in ['VQA','Supervised','Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
        epoch_loss = running_loss / len(dataloaders[phase].dataset)        
    
    """ Removed Recently """
    #representations_list = np.concatenate(representations_list)
    
    if goal == 'IC':
        """ BLEU Score Evaluation """
#        epoch_bleu = calculate_bleu_score(outputs_dict,labels_dict,token2id_dict)
#        epoch_rouge = calculate_rouge_score(outputs_dict,labels_dict,token2id_dict)
#        epoch_meteor = calculate_meteor_score(outputs_dict,labels_dict,token2id_dict)        
        return epoch_loss, epoch_bleu, epoch_rouge, epoch_meteor, outputs_dict, labels_dict #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
    elif goal == 'VQA':
        """ Accuracy of Answers """
        epoch_acc = calculate_answer_accuracy(outputs_dict,class_labels_dict)
        return epoch_loss, epoch_acc  #representations_list, labels_list #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
    elif goal in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection']:
        if goal in ['Language_Change_Detection','Language_Detection']:
            epoch_acc = calculate_language_detection_accuracy(class_predictions_dict,class_labels_dict,goal)
        else:
            """ Accuracy of Answers """
            epoch_acc = calculate_answer_accuracy(class_predictions_dict,class_labels_dict)
        return epoch_loss, epoch_acc #representations_list, labels_list #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
    elif goal in ['MLM','ELECTRA','MARGE']:
        return epoch_loss#, outputs_dict, labels_dict #representations_list, labels_list #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
        

def one_epoch_finetuning(weighted_sampling,phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,criterion,classification,bptt_steps=0,auxiliary_loss=False,attn_loss=False,epoch_count=None,new_task_epochs=None,trial=None,goal='IC',save_path_dir=None):
    if goal == 'IC': #labels_list, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
        epoch_loss, epoch_bleu, epoch_rouge, epoch_meteor, outputs_dict, labels_dict = finetuning_single(phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,weighted_sampling,criterion,classification,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,epoch_count=epoch_count,new_task_epochs=new_task_epochs,trial=trial,goal=goal,save_path_dir=save_path_dir)
        return {"epoch_loss": epoch_loss, "epoch_bleu": epoch_bleu, "epoch_rouge": epoch_rouge, "epoch_meteor": epoch_meteor}, outputs_dict, labels_dict #, labels_list, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
    elif goal in ['VQA','Supervised','Text_Supervised','Language_Change_Detection','Language_Detection']: #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
        epoch_loss, epoch_acc = finetuning_single(phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,weighted_sampling,criterion,classification,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,epoch_count=epoch_count,new_task_epochs=new_task_epochs,trial=trial,goal=goal,save_path_dir=save_path_dir)
        return {"epoch_loss": epoch_loss, "epoch_acc": epoch_acc} #, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
    elif goal in ['MLM','ELECTRA','MARGE']:
        epoch_loss = finetuning_single(phase,token2id_dict,id2embedding_dict,inference,dataloader,model,optimizer,device,weighted_sampling,criterion,classification,auxiliary_loss=auxiliary_loss,attn_loss=attn_loss,epoch_count=epoch_count,new_task_epochs=new_task_epochs,trial=trial,goal=goal,save_path_dir=save_path_dir)
        return {"epoch_loss": epoch_loss}#, outputs_dict, labels_dict#, modality_list, indices_list, task_names_list, class_labels_list, attn_coefs_list, sentence_lens_list
        



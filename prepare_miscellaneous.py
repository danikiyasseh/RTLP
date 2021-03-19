#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:26:21 2020

@author: Dani Kiyasseh
"""
import pickle
import os
import torch.nn as nn
import torch
import numpy as np
import copy
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from scipy.special import expit
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from operator import itemgetter
import torchtext
from rouge_score import rouge_scorer
from spacy_pipeline_101 import modify_text
#%%
""" Functions in this script:
    1) flatten_arrays
    2) obtain_contrastive_loss
    3) calculate_auc
    4) change_labels_type
    5) print_metrics
    6) save_metrics
    7) track_metrics
    8) save_config_weights
    9) save_patient_representations
    10) determine_classification_setting
    11) modify_dataset_order_for_multi_task_learning
    12) obtain_saved_weights_name
    13) make_dir
    14) make_saving_directory
    15) obtain_information
    16) obtain_criterion
"""
#%%

def flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list):
    outputs_list = np.concatenate(outputs_list)
    labels_list = np.concatenate(labels_list)
    modality_list = np.concatenate(modality_list)
    indices_list = np.concatenate(indices_list)
    task_names_list = np.concatenate(task_names_list)
    pids_list = np.concatenate(pids_list)
    return outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

def obtain_BYOL_loss(online_network,target_network,inputs):
    """ Calculate BYOL Loss 
    Args:
        model (nn.Module):
        inputs (torch.tensor): Bx1xSx2 (two views)
    Returns:
        loss (torch.tensor)
    """
    criterion = nn.MSELoss()
    """ Obtain Two Views """
    view1 = inputs[:,:,:,0].unsqueeze(-1) #unsqueeze is just to satisfy how the model expects inputs
    view2 = inputs[:,:,:,1].unsqueeze(-1) #unsqueeze is just to satisfy how the model expects inputs
    """ Obtain Two Models """
    output1 = online_network(view1,BYOL_prediction_head=True) #only applied to online network
    output2 = target_network(view2)
    """ Normalize Outputs """
    norm_output1 = output1.norm(p=2,dim=1).unsqueeze(1)
    output1 = output1/norm_output1
    norm_output2 = output2.norm(p=2,dim=1).unsqueeze(1)
    output2 = output2/norm_output2
    """ Calculate Loss """
    loss1 = criterion(output1,output2)

    """ Symmetric Loss Component """
    """ Obtain Two Models """
    output1 = online_network(view2,BYOL_prediction_head=True) #only applied to online network
    output2 = target_network(view1)
    """ Normalize Outputs """
    norm_output1 = output1.norm(p=2,dim=1).unsqueeze(1)
    output1 = output1/norm_output1
    norm_output2 = output2.norm(p=2,dim=1).unsqueeze(1)
    output2 = output2/norm_output2
    """ Calculate Loss """
    loss2 = criterion(output1,output2) 
    
    """ Total Loss """
    loss = loss1 + loss2
    return loss

def obtain_contrastive_loss(latent_embeddings,pids,trial):
    """ Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
        pids (list): patient ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    """
    if trial in ['CMSC','CMLC','CMSMLC']:
        pids = np.array(pids,dtype=np.object)   
        pid1,pid2 = np.meshgrid(pids,pids)
        pid_matrix = pid1 + '-' + pid2
        pids_of_interest = np.unique(pids + '-' + pids) #unique combinations of pids of interest i.e. matching
        bool_matrix_of_interest = np.zeros((len(pids),len(pids)))
        for pid in pids_of_interest:
            bool_matrix_of_interest += pid_matrix == pid
        rows1,cols1 = np.where(np.triu(bool_matrix_of_interest,1))
        rows2,cols2 = np.where(np.tril(bool_matrix_of_interest,-1))

    nviews = set(range(latent_embeddings.shape[2]))
    view_combinations = combinations(nviews,2)
    loss = 0
    ncombinations = 0
    for combination in view_combinations:
        view1_array = latent_embeddings[:,:,combination[0]] #(BxH)
        view2_array = latent_embeddings[:,:,combination[1]] #(BxH)
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
        norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
        temperature = 0.1
        argument = sim_matrix/(norm_matrix*temperature)
        sim_matrix_exp = torch.exp(argument)
        
        if trial == 'CMC':
            """ Obtain Off Diagonal Entries """
            #upper_triangle = torch.triu(sim_matrix_exp,1)
            #lower_triangle = torch.tril(sim_matrix_exp,-1)
            #off_diagonals = upper_triangle + lower_triangle
            diagonals = torch.diag(sim_matrix_exp)
            """ Obtain Loss Terms(s) """
            loss_term1 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,1)))
            loss_term2 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,0)))
            loss += loss_term1 + loss_term2 
            loss_terms = 2
        elif trial == 'SimCLR':
            self_sim_matrix1 = torch.mm(view1_array,view1_array.transpose(0,1))
            self_norm_matrix1 = torch.mm(norm1_vector.transpose(0,1),norm1_vector)
            temperature = 0.1
            argument = self_sim_matrix1/(self_norm_matrix1*temperature)
            self_sim_matrix_exp1 = torch.exp(argument)
            self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1,1) + torch.tril(self_sim_matrix_exp1,-1)
            
            self_sim_matrix2 = torch.mm(view2_array,view2_array.transpose(0,1))
            self_norm_matrix2 = torch.mm(norm2_vector.transpose(0,1),norm2_vector)
            temperature = 0.1
            argument = self_sim_matrix2/(self_norm_matrix2*temperature)
            self_sim_matrix_exp2 = torch.exp(argument)
            self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2,1) + torch.tril(self_sim_matrix_exp2,-1)

            denominator_loss1 = torch.sum(sim_matrix_exp,1) + torch.sum(self_sim_matrix_off_diagonals1,1)
            denominator_loss2 = torch.sum(sim_matrix_exp,0) + torch.sum(self_sim_matrix_off_diagonals2,0)
            
            diagonals = torch.diag(sim_matrix_exp)
            loss_term1 = -torch.mean(torch.log(diagonals/denominator_loss1))
            loss_term2 = -torch.mean(torch.log(diagonals/denominator_loss2))
            loss += loss_term1 + loss_term2
            loss_terms = 2
        elif trial in ['CMSC','CMLC','CMSMLC']: #ours #CMSMLC = positive examples are same instance and same patient
            triu_elements = sim_matrix_exp[rows1,cols1]
            tril_elements = sim_matrix_exp[rows2,cols2]
            diag_elements = torch.diag(sim_matrix_exp)
            
            triu_sum = torch.sum(sim_matrix_exp,1)
            tril_sum = torch.sum(sim_matrix_exp,0)
            
            loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
            
            loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
            
            loss = loss_diag1 + loss_diag2
            loss_terms = 2

            if len(rows1) > 0:
                loss += loss_triu #technically need to add 1 more term for symmetry
                loss_terms += 1
            
            if len(rows2) > 0:
                loss += loss_tril #technically need to add 1 more term for symmetry
                loss_terms += 1
        
            #print(loss,loss_triu,loss_tril)

        ncombinations += 1
    loss = loss/(loss_terms*ncombinations)
    return loss

def obtain_auto_encoder_loss(outputs,inputs):
    criterion = nn.MSELoss()
    loss = criterion(outputs,inputs)
    return loss

def calculate_auc(classification,outputs_list,labels_list,save_path_dir):
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    if classification is not None:
        if classification != '2-way':
            all_auc = []
            for i in range(labels_ohe.shape[1]):
                auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                all_auc.append(auc)
            epoch_auroc = np.mean(all_auc)
        elif classification == '2-way':
            if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir:
                """ Use This for MultiLabel Process -- Only for Physionet2020 """
                all_auc = []
                for i in range(labels_ohe.shape[1]):
                    auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                    all_auc.append(auc)
                epoch_auroc = np.mean(all_auc)
            else:
                epoch_auroc = roc_auc_score(labels_list,outputs_list)
    else:
        print('This is not a classification problem!')
    return epoch_auroc

#def calculate_acc(outputs_list,labels_list,save_path_dir):
#    if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir: #multilabel scenario
#        """ Convert Preds to Multi-Hot Vector """
#        preds_list = np.where(outputs_list>0.5,1,0)
#        """ Indices of Hot Vectors of Predictions """
#        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
#        """ Indices of Hot Vectors of Ground Truth """
#        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
#        """ What Proportion of Labels Did you Get Right """
#        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
#    else: #normal single label setting 
#        preds_list = torch.argmax(torch.tensor(outputs_list),1)
#        ncorrect_preds = (preds_list == torch.tensor(labels_list)).sum().item()
#        acc = ncorrect_preds/preds_list.shape[0]
#    return acc

def calculate_acc(classification,outputs_list,labels_list,pids_list,save_path_dir,eval_level='instance'):
    if 'physionet2020' in save_path_dir:# or 'ptbxl' in save_path_dir: #multilabel scenario
        """ Convert Outputs to Preds """
        preds_list = expit(outputs_list)
        """ Convert Preds to Multi-Hot Vector """
        preds_list = np.where(preds_list>0.5,1,0)
        """ Indices of Hot Vectors of Predictions """
        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
        """ Indices of Hot Vectors of Ground Truth """
        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
        """ What Proportion of Labels Did you Get Right """
        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
    elif classification == '2-way':
        preds_list = expit(outputs_list)        
        preds_list = torch.tensor(np.where(preds_list>0.5,1,0))
        preds_list = preds_list.squeeze()
        labels_list = torch.tensor(labels_list)
        if eval_level == 'patient':
            acc = obtain_patient_level_accuracy(pids_list,preds_list,labels_list)
        else:
            print(preds_list.shape,labels_list.shape)
            ncorrect_preds = (preds_list == labels_list).sum().item()
            acc = ncorrect_preds/preds_list.shape[0]        
    else: #normal single label setting 
        preds_list = torch.softmax(torch.tensor(outputs_list),1)
        preds_list = torch.argmax(preds_list,1)
        labels_list = torch.tensor(labels_list)
        if eval_level == 'patient':
            acc = obtain_patient_level_accuracy(pids_list,preds_list,labels_list)
        else:        
            ncorrect_preds = (preds_list == labels_list).sum().item()
            acc = ncorrect_preds/preds_list.shape[0]
    return acc

def obtain_patient_level_accuracy(pids_list,preds_list,labels_list):
    patient_preds_list = []
    patient_labels_list = []
    """ Obtain Unique PIDs """
    unique_pids = np.unique(pids_list)
    """ Iterate Through PIDs """
    for pid in unique_pids:
        """ Return Indices of Current PID """
        indices = np.where(np.in1d(pids_list,pid))[0]
        """ Return Set of Preds List of Current PID """
        pid_preds = preds_list[indices]
        """ Return Most Common Prediction ----- This Determines How to Aggregate Instance Predictions """
        value,index = pid_preds.mode()
        pid_pred = value.item()
        """ Obtain Ground Truth Scalar - Just Choose One """
        pid_label = labels_list[indices][0].item()
        """ Append New Patient-Level Values """
        patient_preds_list.append(pid_pred)
        patient_labels_list.append(pid_label)
    """ Convert to Tensor """
    patient_preds_list = torch.tensor(patient_preds_list)
    patient_labels_list = torch.tensor(patient_labels_list)
    """ Calculate Patient-Level Accuracy """
    ncorrect_preds = (patient_preds_list == patient_labels_list).sum().item()
    acc = ncorrect_preds/patient_preds_list.shape[0]   
    return acc  

#def aggregate_results(text_indices):
#    """ These Need to be Language Specific """
#    for dest_lang in text_indices.keys():
#
#        current_sentence_lens = sentence_lens[dest_lang]
#        current_class_labels = current_class_labels_dict[dest_lang]
#        current_class_predictions = current_class_predictions_dict[dest_lang]
#
#        if goal in ['Language_Change_Detection','Language_Detection']:
#            current_class_labels = [labels.cpu().detach().numpy() for labels in current_class_labels]
#            current_class_predictions = [predictions.cpu().detach().numpy() for predictions in current_class_predictions]
#        else:
#            current_class_labels = current_class_labels.cpu().detach().numpy()
#            current_class_predictions = current_class_predictions.cpu().detach().numpy()
#        
#        """ Store Batch Data in The Dictionaries """
#        sentence_lens_dict[dest_lang].extend(current_sentence_lens)
#        class_labels_dict[dest_lang].extend(current_class_labels) #.cpu().detach().numpy())
#        class_predictions_dict[dest_lang].extend(current_class_predictions) #.cpu().detach().numpy())
#
#        if goal not in ['Supervised','Text_Supervised','Language_Change_Detection','Language_Detection']:
#            if current_labels_dict[dest_lang].data.dtype != torch.long:
#                current_labels_dict[dest_lang].data = current_labels_dict[dest_lang].data.type(torch.long)
#
#            current_text_indices = current_labels_dict[dest_lang]
#            current_outputs = current_outputs_dict[dest_lang]
#            current_attn_coefs = current_attn_coefs_dict[dest_lang]
#            current_representations = current_representations_dict[dest_lang]
#            """ Store Batch Data in The Dictionaries """ 
#            labels_dict[dest_lang].extend(current_text_indices.cpu().detach().numpy())
#            outputs_dict[dest_lang].extend(current_outputs.cpu().detach().numpy())
#            attn_coefs_dict[dest_lang].extend(current_attn_coefs.cpu().detach().numpy())
#            representations_dict[dest_lang].extend(current_representations.cpu().detach().numpy())
#        elif goal in ['Text_Supervised']:
#            current_representations = current_representations_dict[dest_lang]
#            representations_dict[dest_lang].extend(current_representations.squeeze().cpu().detach().numpy())        
#        else:
#            current_representations = current_representations_dict[dest_lang]
#            if goal in ['Language_Change_Detection','Language_Detection']:
#                current_representations = [representations.cpu().detach().numpy() for representations in current_representations]
#            else:
#                current_representations = current_representations.cpu().detach().numpy()
#            representations_dict[dest_lang].extend(current_representations)

#%%
def calculate_IC_loss(criterion,outputs,text_indices,class_predictions,class_labels,attn_coefs,auxiliary_loss,attn_loss):
    #""" Sort the Target Indices According to Packing Procedure Sorting *** IMPORTANT *** """
    #text_indices = text_indices.index_select(0,sorted_indices) #B x Words
    #print(outputs,text_indices)
    """ Calculate Loss At the Word/Token Level """
    loss = criterion(outputs.permute(0,2,1),text_indices) #outputs has to be BxC(classes)xHxW i.e., B x Words x S
    print(loss)
    """ Calculate Auxiliary Class Loss """
    if auxiliary_loss == True:
        class_prediction_criterion = nn.CrossEntropyLoss()
        class_prediction_loss = class_prediction_criterion(class_predictions,class_labels)
        loss = loss + class_prediction_loss
    """ Calculate Attention Coefs Loss """
    if attn_loss == True:
        average_over_sequence = torch.pow((1 - torch.sum(attn_coefs,dim=1)),2) # BxL
        average_over_annot_vectors = torch.sum(average_over_sequence,dim=1) # B
        attn_coefs_loss = torch.mean(average_over_annot_vectors)
        lambda_attn_loss = 0.01
        loss = loss + (lambda_attn_loss * attn_coefs_loss) 
    return loss


def strip_output_tokens(sentence,end_token):
    """ Remove All Tokens Including and Beyond '/END' """
#    if '/END' in sentence:
    #print(sentence)
    if end_token.lower() in sentence:
        try: 
            sentence = sentence[:np.where(np.in1d(sentence,[end_token.lower()]))[0].item()] 
        except: 
            sentence
    else:
        sentence
    return sentence

def strip_target_tokens(sentence,end_token):
    #print(sentence)
    #print(end_token.lower())
    """ Remove All Tokens Including and Beyond '/END' """
    #print(sentence)
    #sentence = sentence[1:np.where(np.in1d(sentence,[end_token.lower()]))[0].item()] #guaranteed to exist since target/ground-truth always has END
    """ Modified January 27, 2021 (Remove 1) """
    sentence = sentence[:np.where(np.in1d(sentence,[end_token.lower()]))[0].item()] #guaranteed to exist since target/ground-truth always has END
    return sentence

def calculate_bleu_score(output_dict,label_dict,token2id_dict):
    language_bleu_scores = dict()
    for dest_lang in token2id_dict.keys():
        #print(dest_lang)
        output_probs = output_dict[dest_lang]
        target_indices = label_dict[dest_lang]
        current_token2id_dict = token2id_dict[dest_lang]
        """ Obtain Language to Start and End Token Mapping """
        text_modifier = modify_text(dest_lang,'filler')
        start_dict,end_dict = text_modifier.obtain_start_and_end_dict()
        """ End Token """
        end_token = end_dict[dest_lang]
        """ Obtain ID to Token Mapping """
        id2token_dict = {value:key for key,value in current_token2id_dict.items()}  
        """ Convert Output Probabilities to Indices """
        #output_indices = [np.argmax(sentence_probs,-1) for batch_probs in output_probs for sentence_probs in batch_probs]
        #""" NEW - DECEMBER 22 - ADDED -1 AFTER ARGMAX """
        output_indices = [np.argmax(sentence_probs,-1) for sentence_probs in output_probs]
        """ Obtain Tokens From IDs """
        #output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        """ NEW - DECEMBER 31 2020 - Ensure Predictions Are Within Bounds """
        output_tokens = [[id2token_dict[index] if index < len(id2token_dict)-1 else id2token_dict[0] for index in sentence_indices] for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        target_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in target_indices]# for sentence_indices in batch_indices]
        """ Remove Tokens Beyond /END (If End is Available) """
        output_tokens = [strip_output_tokens(sentence,end_token) for sentence in output_tokens]
        """ Remove Tokens Beyond /END AND Remove '/START' Token """    
        target_tokens = [strip_target_tokens(sentence,end_token) for sentence in target_tokens]
        """ Calculate BLEU Score For Each Sentence """ #0.18 unigram
        bleu_scores = [sentence_bleu([target_sentence_token],output_sentence_token,weights=[1]) for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        #bleu_scores = [torchtext.data.metrics.bleu_score([output_sentence_token],[[target_sentence_token]]) for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        """ Calculate Average BLEU Score """
        bleu_score = np.mean(bleu_scores)
        """ Store As Language-Specific Score """
        language_bleu_scores[dest_lang] = bleu_score
    return language_bleu_scores

def calculate_rouge_score(output_dict,label_dict,token2id_dict):
    language_rouge_scores = dict()
    for dest_lang in token2id_dict.keys():
        output_probs = output_dict[dest_lang]
        target_indices = label_dict[dest_lang]
        current_token2id_dict = token2id_dict[dest_lang]    
        """ Obtain Language to Start and End Token Mapping """
        text_modifier = modify_text(dest_lang,'filler')
        start_dict,end_dict = text_modifier.obtain_start_and_end_dict()
        """ End Token """
        end_token = end_dict[dest_lang]
        """ Obtain ID to Token Mapping """
        id2token_dict = {value:key for key,value in current_token2id_dict.items()}  
        """ Convert Output Probabilities to Indices """
        #output_indices = [np.argmax(sentence_probs,-1) for batch_probs in output_probs for sentence_probs in batch_probs]
        output_indices = [np.argmax(sentence_probs,-1) for sentence_probs in output_probs]
        """ Obtain Tokens From IDs """
        #output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in output_indices]
        #output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        output_tokens = [[id2token_dict[index] if index < len(id2token_dict)-1 else id2token_dict[0] for index in sentence_indices] for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        target_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in target_indices]# for sentence_indices in batch_indices]
        """ Remove Tokens Beyond /END (If End is Available) """
        output_tokens = [strip_output_tokens(sentence,end_token) for sentence in output_tokens]
        """ Remove Tokens Beyond /END AND Remove '/START' Token """    
        target_tokens = [strip_target_tokens(sentence,end_token) for sentence in target_tokens] 
        """ Calculate Rouge for Each Sentence """
        scorer = rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
        """ Obtain Fmeasure From Rouge Scores """  #expects list of strings and list of strings
        rouge_scores = [scorer.score(' '.join(target_sentence_token),' '.join(output_sentence_token))['rougeL'][-1] for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        #bleu_scores = [torchtext.data.metrics.bleu_score([output_sentence_token],[[target_sentence_token]]) for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        """ Calculate Average BLEU Score """
        rouge_score = np.mean(rouge_scores)
        """ Store As Language-Specific Score """
        language_rouge_scores[dest_lang] = rouge_score
    return language_rouge_scores

def calculate_meteor_score(output_dict,label_dict,token2id_dict):
    language_meteor_scores = dict()
    for dest_lang in token2id_dict.keys():
        output_probs = output_dict[dest_lang]
        target_indices = label_dict[dest_lang]
        current_token2id_dict = token2id_dict[dest_lang]    
        """ Obtain Language to Start and End Token Mapping """
        text_modifier = modify_text(dest_lang,'filler')
        start_dict,end_dict = text_modifier.obtain_start_and_end_dict()
        """ End Token """
        end_token = end_dict[dest_lang]
        """ Obtain ID to Token Mapping """
        id2token_dict = {value:key for key,value in current_token2id_dict.items()}  
        """ Convert Output Probabilities to Indices """
        #output_indices = [np.argmax(sentence_probs,-1) for batch_probs in output_probs for sentence_probs in batch_probs]
        output_indices = [np.argmax(sentence_probs,-1) for sentence_probs in output_probs]
        """ Obtain Tokens From IDs """
        #output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in output_indices]
        #output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        output_tokens = [[id2token_dict[index] if index < len(id2token_dict)-1 else id2token_dict[0] for index in sentence_indices] for sentence_indices in output_indices]# for sentence_indices in batch_indices]
        target_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in target_indices]# for sentence_indices in batch_indices]
        """ Remove Tokens Beyond /END (If End is Available) """
        output_tokens = [strip_output_tokens(sentence,end_token) for sentence in output_tokens]
        """ Remove Tokens Beyond /END AND Remove '/START' Token """    
        target_tokens = [strip_target_tokens(sentence,end_token) for sentence in target_tokens] 
        """ Calculate Meteor Score for Each Sentence """ #
        scores = [meteor_score([' '.join(target_sentence_token)],' '.join(output_sentence_token)) for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        #bleu_scores = [torchtext.data.metrics.bleu_score([output_sentence_token],[[target_sentence_token]]) for output_sentence_token,target_sentence_token in zip(output_tokens,target_tokens)]
        """ Calculate Average BLEU Score """
        mean_score = np.mean(scores)
        """ Store As Language-Specific Score """
        language_meteor_scores[dest_lang] = mean_score        
    return language_meteor_scores

def convert_predicted_ids_to_sentences(batch_indices,token2id_dict,dest_lang):
    id2token_dict = {value:key for key,value in token2id_dict.items()}  
    """ Convert Output Probabilities to Indices """
    #output_indices = [np.argmax(batch_indices,-1) for batch_indices in output_indices]
    batch_indices = np.argmax(batch_indices,-1)
    """ Obtain Tokens From IDs """
    output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in batch_indices]# for sentence_indices in batch_indices]
    """ Obtain Language to Start and End Token Mapping """
    text_modifier = modify_text(dest_lang,'filler')
    start_dict,end_dict = text_modifier.obtain_start_and_end_dict()
    """ End Token """
    end_token = end_dict[dest_lang]
    """ Remove Tokens Beyond /END (If End is Available) """
    output_tokens = [strip_output_tokens(sentence,end_token) for sentence in output_tokens]
    #print(output_tokens)
    return output_tokens    

def save_predicted_sentences(save_path_dir,output_indices,token2id_dict):
    id2token_dict = {value:key for key,value in token2id_dict.items()}  
    """ Convert Output Probabilities to Indices """
    output_indices = [np.argmax(batch_indices,-1) for batch_indices in output_indices]
    """ Obtain Tokens From IDs """
    output_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for batch_indices in output_indices for sentence_indices in batch_indices]
    """ Remove Tokens Beyond /END (If End is Available) """
    output_tokens = [strip_output_tokens(sentence) for sentence in output_tokens]
    """ Save Sentences """
    torch.save(output_tokens,os.path.join(save_path_dir,'output_sentences'))

def convert_target_ids_to_sentences(batch_indices,token2id_dict,dest_lang):
    id2token_dict = {value:key for key,value in token2id_dict.items()}  
    """ Obtain Tokens From IDs """
    target_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for sentence_indices in batch_indices]# for sentence_indices in batch_indices]
    """ Obtain Language to Start and End Token Mapping """
    text_modifier = modify_text(dest_lang,'filler')
    start_dict,end_dict = text_modifier.obtain_start_and_end_dict()
    """ End Token """
    end_token = end_dict[dest_lang]
    """ Remove Tokens Beyond /END (If End is Available) """
    target_tokens = [strip_target_tokens(sentence,end_token) for sentence in target_tokens]
    return target_tokens

def save_target_sentences(save_path_dir,target_indices,token2id_dict):
    id2token_dict = {value:key for key,value in token2id_dict.items()}  
    """ Obtain Tokens From IDs """
    target_tokens = [list(itemgetter(*sentence_indices.tolist())(id2token_dict)) for batch_indices in target_indices for sentence_indices in batch_indices]
    """ Remove Tokens Beyond /END (If End is Available) """
    target_tokens = [strip_target_tokens(sentence) for sentence in target_tokens]
    """ Save Sentences """
    torch.save(target_tokens,os.path.join(save_path_dir,'target_sentences'))

def calculate_language_detection_accuracy(output_dict,label_dict,goal):
    language_accuracy = dict()
    for dest_lang in output_dict.keys():
        outputs = output_dict[dest_lang]
        labels = label_dict[dest_lang]
        #np.save('output_dict',output_dict)
        #np.save('label_dict',label_dict)
        total_ncorrect_preds = 0
        total_entries = 0
        for sentence_output_predictions,sentence_labels in zip(outputs,labels):
            if goal == 'Language_Change_Detection':
                predictions = np.where(expit(np.array(sentence_output_predictions)) > 0.5,1,0)
            elif goal == 'Language_Detection':
                predictions = np.argmax(sentence_output_predictions,1)

            ncorrect_preds = np.sum(predictions == sentence_labels)
            nentries = len(predictions)
            
            total_ncorrect_preds += ncorrect_preds
            total_entries += nentries

        accuracy = total_ncorrect_preds / total_entries
        """ Store As Language-Specifiic Accuracy """
        language_accuracy[dest_lang] = accuracy
    return language_accuracy


def calculate_answer_accuracy(output_dict,label_dict):
    language_accuracy = dict()
    for dest_lang in output_dict.keys():
        outputs = output_dict[dest_lang]
        labels = label_dict[dest_lang]
        """ Calculate Accuracy of Answers in VQA Setting """
        outputs = np.concatenate(outputs) # NxC
        labels = np.concatenate(labels) # N 
        #np.save('outputsss',outputs)
        #np.save('labelsss',labels)
        predictions = np.argmax(outputs,1) #argmax or argmax of softmax will lead to same result so no biggie if you dont include softmax
        ncorrect_preds = np.sum(predictions == labels)
        accuracy = ncorrect_preds/len(predictions)
        """ Store As Language-Specifiic Accuracy """
        language_accuracy[dest_lang] = accuracy
    return language_accuracy

def change_labels_type(labels,criterion):
    if isinstance(criterion,nn.BCEWithLogitsLoss):
        labels = labels.type(torch.float)
    elif isinstance(criterion,nn.CrossEntropyLoss):
        labels = labels.type(torch.long)
    return labels

def print_metrics(phase,results_dictionary):
    metric_name_to_label = {'epoch_loss':'loss','epoch_bleu':'bleu','epoch_auroc':'auc','epoch_acc':'acc','epoch_rouge':'rouge','epoch_meteor':'meteor'}
    items_to_print = dict()
    labels = []
    for metric_name,result in results_dictionary.items():
        if metric_name in ['epoch_acc','epoch_bleu','epoch_rouge','epoch_meteor']:
            for dest_lang,lang_result in result.items():
                label_suffix = metric_name_to_label[metric_name]
                label = '-'.join((phase,dest_lang,label_suffix))
                labels.append(label)
                items_to_print[label] = ['%.4f' % lang_result]
        else:
            label_suffix = metric_name_to_label[metric_name]
            label = '-'.join((phase,label_suffix))
            labels.append(label)
            items_to_print[label] = ['%.4f' % result]
    print(tabulate(items_to_print,labels))

def save_metrics(save_path_dir,prefix,metrics_dict):
    torch.save(metrics_dict,os.path.join(save_path_dir,'%s_metrics_dict' % prefix))

def track_metrics(metrics_dict,results_dictionary,phase,epoch_count):
    for metric_name,results in results_dictionary.items():
        
        if epoch_count == 0 and ('train' in phase or 'test' in phase):
            metrics_dict[metric_name] = dict()
        
        if epoch_count == 0:
            metrics_dict[metric_name][phase] = []
        
        metrics_dict[metric_name][phase].append(results)
    return metrics_dict

def save_config_weights(save_path_dir,best_model_weights,saved_weights_name,phases,trial,downstream_dataset): #which is actually second_dataset
    if trial in ['Linear','Fine-Tuning','Random']:
        saved_weights_name = 'finetuned_weight'
    torch.save(best_model_weights,os.path.join(save_path_dir,saved_weights_name))

def save_patient_representation(save_path_dir,patient_rep_dict,trial):
    if trial not in ['Linear','Fine-Tuning']:
        with open(os.path.join(save_path_dir,'patient_rep'),'wb') as f:
            pickle.dump(patient_rep_dict,f)

def determine_classification_setting(dataset_name,trial,output_type):
    #dataset_name = dataset_name[0]
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2016':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '2-way' #because binary multilabel
    elif dataset_name == 'chapman':
        classification = '4-way'
    elif dataset_name == 'chapman_pvc':
        classification = '2-way'
    elif dataset_name == 'ptbxl':
        if output_type == 'single':
            classification = '5-way' #5-way
        elif output_type == 'multi':
            classification = '2-way'
    else: #used for pretraining with contrastive learning
        classification = None
    #print('Original Classification %s' % classification)
    return classification

def modify_dataset_order_for_multi_task_learning(dataset,modalities,leads,class_pairs,fractions):
    dataset = [dataset] #outside of if statement because dataset is original during each iteration
    if not isinstance(fractions,list): #it is already in list format, therefore no need for extra list
        modalities = [modalities]
        leads = [leads]
        class_pairs = [class_pairs]
        fractions = [fractions]
    return dataset,modalities,leads,class_pairs,fractions

def obtain_saved_weights_name(trial_to_run,trial_to_load,phases):
    if trial_to_load in ['Random'] and trial_to_run in ['Linear','Fine-Tuning']: #supervised pre-training transfer
        saved_weights = 'finetuned_weight' #name of weights to load and save (in diff directories though)
    elif trial_to_run not in ['Linear','Fine-Tuning','Random']: #self-supervised pre-training 
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to save 
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'pretrained_weight' #name of weights to load
    elif trial_to_run in ['Linear','Fine-Tuning','Random']: #downstream task training
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to load
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'finetuned_weight' #name of weights to load
    return saved_weights

def obtain_load_path_dir(phases,save_path_dir,trial_to_load,trial_to_run,second_input_perturbed,second_perturbation,second_dataset,labelled_fraction,leads,max_seed,task,evaluation=False):
    """ This is Used to Add Suffix to Save Path Dir According to Second Parameters """
    if trial_to_run in ['Linear','Fine-Tuning','Random']:
        labelled_fraction_path = 'training_fraction_%.2f' % labelled_fraction
        leads_path = 'leads_%s' % str(leads[0]) #remember leads is a list of lists 
        if trial_to_run in ['Random']:
            trial_to_run = ''
            if second_dataset in ['chapman','physionet2020']:
                leads_path = 'leads_%s' % str(leads[0]) #only these two datasets have multiple leads
            else:
                leads_path = ''

        if leads[0] == None:
            leads_path = ''
            
        if second_input_perturbed == True:
            perturbed_path = 'perturbed'
            perturbation_path = str(second_perturbation)
        elif second_input_perturbed == False:
            perturbed_path = ''
            perturbation_path = ''
        
        save_path_dir = os.path.join(save_path_dir,trial_to_run,second_dataset,leads_path,perturbed_path,perturbation_path,labelled_fraction_path)        
        #print(save_path_dir)
        if 'train' in phases:
            save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=True,evaluation=evaluation) #do NOT change second_pass = True b/c this function is only ever used during second pass
        elif 'test' in phases:
            if 'test_metrics_dict' in os.listdir(save_path_dir):
                save_path_dir = 'do not test'
        
        if save_path_dir in ['do not train','do not test']:
            load_path_dir = save_path_dir
        else:
            if trial_to_load in ['Random'] and trial_to_run in ['Linear','Fine-Tuning']: #transfer from supervised pre-training to downstream
                load_path_dir = save_path_dir.split(trial_to_run)[0]
            else:
                split_save_path_dir = save_path_dir.split('/')
                seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
                load_path_dir = '/'.join(split_save_path_dir[:seed_index+1]) #you want to exclude everything AFTER the seed path
    else:
        load_path_dir = save_path_dir

    if evaluation == False:
        print('Load Path Dir: %s' % load_path_dir)
        print('Save Path Dir: %s' % save_path_dir)
    
    return load_path_dir, save_path_dir

def make_saving_directory(phases,dataset_name,trial_to_load,trial_to_run,seed,max_seed,task,BYOL_tau,embedding_dim,leads,input_perturbed,perturbation,band_types,nbands,band_width,
                          encoder_type,decoder_type,attn,attn_loss,goal,pretrained_embeddings,freeze_embeddings,pretrained_encoder,freeze_encoder,pretrained_decoder,freeze_decoder,
                          dest_langs_list,target_token_selection,replacement_prob,num_layers,evaluation=False):
    base_path = '/mnt/SecondaryHDD/PhysioVQA Results' 
    seed_path = 'seed%i' % int(seed)
    dataset_path = dataset_name#[0] #dataset used for training
    if leads is None:
        leads_path = ''
    else:
        leads_path = 'leads_%s' % str(leads) #leads used for training
    embedding_path = 'embedding_%i' % embedding_dim #size of embbedding used
    if trial_to_run in ['Linear','Fine-Tuning']:
        trial_path = trial_to_load
    elif trial_to_run in ['Random']:
        trial_path = trial_to_run
        dataset_path, leads_path = '', ''
    else:
        trial_path = trial_to_run

    if input_perturbed == True:
        perturbed_path = 'perturbed'
        perturbation_path = str(perturbation)
        if 'SpecAugment' in perturbation:
            band_types_path = 'band_types_%s' % str(band_types)
            nbands_path = 'nbands_%s' % str(nbands)
            band_width_path = 'band_width_%s' % str(band_width)
        else:
            band_types_path = ''
            nbands_path = ''
            band_width_path = ''
    elif input_perturbed == False:
        perturbed_path = ''
        perturbation_path = ''
        band_types_path = ''
        nbands_path = ''
        band_width_path = ''    
    
    if trial_to_load in ['BYOL']:
        BYOL_tau_path = 'tau_%.3f' % BYOL_tau
    else:
        BYOL_tau_path = ''
    
    #if model_type == 'cnn_small':
    #    model_path = ''
    #else:
    model_path = encoder_type
    
    attn_path = 'attn_%s' % str(attn)
    attn_loss_path = 'attn_regularization_%s' % str(attn_loss)
    goal_path = goal
    replacement_prob_path = 'replacement_prob_%s' % str(replacement_prob)
    langs_path = 'langs_%s' % str(dest_langs_list)
    target_token_selection_path = 'target_token_selection_%s' % target_token_selection
    num_layers_path = 'num_layers_%s' % str(num_layers)
    
    embeddings_suffix = '_'.join(list(map(lambda x:str(x),list(pretrained_embeddings.values())))) if isinstance(pretrained_embeddings,dict) else pretrained_embeddings
    pretrained_embeddings_path = 'pretrained_embeddings_%s' % embeddings_suffix
    freeze_embeddings_path = 'freeze_embeddings_%s' % str(freeze_embeddings)
    
    encoder_suffix = '_'.join(list(pretrained_encoder.values())) if isinstance(pretrained_encoder,dict) else pretrained_encoder    
    pretrained_encoder_path = 'pretrained_encoder_%s' % encoder_suffix
    freeze_encoder_path = 'freeze_encoder_%s' % str(freeze_encoder)
    
    decoder_suffix = '_'.join(list(map(lambda x:str(x),list(pretrained_decoder.values())))) if isinstance(pretrained_decoder,dict) else pretrained_decoder #this will have pretrained langs in it
    pretrained_decoder_path = 'pretrained_decoder_%s' % decoder_suffix
    freeze_decoder_path = 'freeze_decoder_%s' % str(freeze_decoder)
    
    """ First If Added New September 9th, 2020 """
#    if trial_to_load in ['Random'] and trial_to_run in ['Linear','Fine-Tuning']: #supervised pre-training pathway
#        #dataset_path = ''
#        upstream_labelled_fraction_path = 'training_fraction_%.2f' % 1.00 #for now, this is hardcoded for supervised pre-training 
#        save_path_dir = os.path.join(base_path,trial_path,model_path,embedding_path,seed_path,dataset_path,leads_path,perturbed_path,upstream_labelled_fraction_path)
#    else:
    if goal in ['IC','VQA']:
        save_path_dir = os.path.join(base_path,trial_path,encoder_type,decoder_type,dataset_path,leads_path,embedding_path,perturbed_path,perturbation_path,band_types_path,nbands_path,
                                         band_width_path,BYOL_tau_path,goal_path,pretrained_embeddings_path,freeze_embeddings_path,
                                         pretrained_encoder_path,freeze_encoder_path,pretrained_decoder_path,freeze_decoder_path,
                                         langs_path,attn_path,attn_loss_path,seed_path)
    elif goal in ['Text_Supervised','Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']:
        if goal in ['Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA','MARGE']: #decoder pre-training
            model_path = decoder_type

            save_path_dir = os.path.join(base_path,trial_path,model_path,num_layers_path,dataset_path,leads_path,embedding_path,perturbed_path,perturbation_path,band_types_path,nbands_path,
                                             band_width_path,BYOL_tau_path,goal_path,langs_path,target_token_selection_path,replacement_prob_path,seed_path)        

        elif goal in ['Supervised']: #encoder supervised pre-training
            model_path = encoder_type

            save_path_dir = os.path.join(base_path,trial_path,model_path,dataset_path,leads_path,embedding_path,perturbed_path,perturbation_path,band_types_path,nbands_path,
                                             band_width_path,BYOL_tau_path,goal_path,seed_path)        
        
    if 'train' in phases:
        save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,evaluation=evaluation)
    elif 'test' in phases:
        if 'test_metrics_dict' in os.listdir(save_path_dir):
            if trial_to_load in ['Random'] and trial_to_run in ['Linear','Fine-Tuning']: #supervised pre-training transfer to downstream then test
                save_path_dir = save_path_dir #without this line, then you cannot evaluate because of presence of test metrics in above folder 
            else:
                save_path_dir = 'do not test'
    
    return save_path_dir, seed

def make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=False,evaluation=False): #boolean allows you to overwrite if TRUE 
    """ Recursive Function to Make Sure I do Not Overwrite Previous Seeds """
    split_save_path_dir = save_path_dir.split('/')
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    current_seed = int(split_save_path_dir[seed_index].strip('seed'))
    try:
        if second_pass == False:
            condition = ('obtain_representation' not in task) and (trial_to_run not in ['Linear','Fine-Tuning'])
        elif second_pass == True:
            condition = ('obtain_representation' not in task)

        if condition:# and trial_to_run not in ['Linear','Fine-Tuning']: #do not skip if you need to do finetuning
            os.chdir(save_path_dir)
            if 'train_val_metrics_dict' in os.listdir() and evaluation == False:
                if current_seed < max_seed-1:
                    print('Skipping Seed!')
                    new_seed = current_seed + 1
                    seed_path = 'seed%i' % new_seed
                    save_path_dir = save_path_dir.replace('seed%i' % current_seed,seed_path)
                    save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=second_pass,evaluation=evaluation)
                else:
                    save_path_dir = 'do not train'
    except:
        os.makedirs(save_path_dir)
    
    if os.path.isdir(save_path_dir) == False: #just in case we miss making the directory somewhere
        os.makedirs(save_path_dir)
    
    if current_seed == max_seed:
        current_seed = 0
    
    return save_path_dir, current_seed

def obtain_information(trial,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict,extra_trial=None):
    if trial in ['Random'] and extra_trial in ['Linear','Fine-Tuning']: #supervised pre-training transfer pathway
        training_dataset = downstream_dataset
    elif trial in ['Linear','Fine-Tuning','Random']: #transfer pathway from self-supervised pre-training
        training_dataset = second_dataset
    else:
        training_dataset = downstream_dataset #used for contrastive training 
    leads = data2leads_dict[training_dataset]
    batch_size = data2bs_dict[training_dataset]
    held_out_lr = data2lr_dict[training_dataset]
    class_pair = data2classpair_dict[training_dataset]
    modalities = ['ecg']
    fraction = 1 #1 for chapman, physio2020, and physio2017. Use labelled_fraction for control over fraction of training data used 
    return leads, batch_size, held_out_lr, class_pair, modalities, fraction       

def obtain_criterion(classification):
    if classification == '2-way':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion
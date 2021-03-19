#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:19:32 2020

@author: Dani Kiyasseh
"""

from prepare_network import image_encoder_network, text_decoder_network, \
                            combined_image_captioning_network, vqa_decoder_network, \
                            text_encoder_network, combined_vqa_network, MARGE_decoder_network, \
                            MARGE
import resnet1d
import torch.optim as optim
import torch
import os
from spacy_pipeline_101 import obtain_token_mapping
import numpy as np
from tabulate import tabulate
#%%
""" Functions in this script:
    1) load_initial_model_contrastive
    2) obtain_noutputs
"""

#%%
def freeze_params(model):
    for param in model.parameters():
        param.requires_grad_(False)
    print('Params Frozen!')

def replace_params(model,loaded_params):
    count = 0
    total = 0
#    for (name1,values1),(name2,values2) in zip(model.state_dict().items(),loaded_params.items()):
#        if name1 == name2 and values1.shape == values2.shape:
#            values1.data.copy_(values2)
#            count += 1
#        total += 1

    for name1,values1 in model.state_dict().items():
        if name1 in list(loaded_params.keys()):
            values2 = loaded_params[name1]
            if values1.shape == values2.shape:
                values1.data.copy_(values2)
                count += 1
        total += 1
    print('%i of %i Parameters Replaced' % (count,total))

def load_encoder_params(encoder_type,embedding_dim,pretrained_encoder,save_path_dir):
    """ Load Pretrained Params for the Encoder 
    NOTE: pretrained_encoder will consist of many of the components """
    
    basepath = '/mnt/SecondaryHDD/PhysioVQA Results/Random'
    pretrained_supervision = pretrained_encoder['supervision']
    pretrained_dataset = pretrained_encoder['dataset']
    pretrained_embedding = 'embedding_%s' % str(embedding_dim)
    split_save_path_dir = save_path_dir.split('/')    
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    seed_path = split_save_path_dir[seed_index]
    if pretrained_dataset == 'chapman':
        leads = "leads_['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']"
    elif pretrained_dataset in ['ptbxl','brazil']:
        leads = ''
    
    path = os.path.join(basepath,encoder_type,pretrained_embedding,pretrained_supervision,seed_path,pretrained_dataset,leads,'training_fraction_1.00/finetuned_weight')        
    params = torch.load(path)
    print('Encoder Params Loaded!')        
    return params
        
def load_embeddings(decoder_type,num_layers,embedding_dim,pretrained_embeddings,token2id_dict,save_path_dir,device):#,nlp):
    basepath = '/mnt/SecondaryHDD/PhysioVQA Results/Random'
    num_layers = 'num_layers_%s' % str(num_layers)
    pretrained_embedding = 'embedding_%s' % str(embedding_dim)
    pretrained_supervision = pretrained_embeddings['supervision']
    pretrained_replacement_prob = 'replacement_prob_%s' % str(pretrained_embeddings['replacement_prob'])
    #pretrained_langs = 'langs_%s' % str(pretrained_embeddings['langs'])
    pretrained_langs = 'langs_%s' % str(['de','el','en','es','fr','it','pt'])
    pretrained_target_token_selection = 'target_token_selection_%s' % str(pretrained_embeddings['target_token_selection'])
    
#    if pretrained_supervision == 'Spacy':
#        id2embedding_dict = {tokenid:torch.rand(embedding_dim,requires_grad=True,device=device) if nlp(token)[0].has_vector == False else torch.tensor(nlp(token)[0].vector,requires_grad=True,device=device) for token,tokenid in token2id_dict.items()}
#        print('Spacy Pretrained Word Embeddings Loaded!')
#    else:
    pretrained_dataset = pretrained_embeddings['dataset']    
    split_save_path_dir = save_path_dir.split('/')    
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    seed_path = split_save_path_dir[seed_index]
    embedding_path = os.path.join(basepath,decoder_type,num_layers,pretrained_embedding,pretrained_supervision,pretrained_langs,pretrained_target_token_selection,pretrained_replacement_prob,seed_path,pretrained_dataset,'training_fraction_1.00/id2embedding_dict')
    id2embedding_dict = torch.load(embedding_path)
    print('Pretrained Word Embeddings Loaded!')
    return id2embedding_dict

def load_decoder_params(decoder_type,num_layers,embedding_dim,pretrained_decoder,save_path_dir,pretraining_task):
    """ Load Pretrained Params for the Decoder 
    NOTE: pretrained_decoder will consist of many components """
    
    basepath = '/mnt/SecondaryHDD/PhysioVQA Results/Random'
    pretrained_supervision = pretrained_decoder['supervision']
    pretrained_replacement_prob = 'replacement_prob_%s' % str(pretrained_decoder['replacement_prob'])
    pretrained_dataset = pretrained_decoder['dataset']
    #pretrained_langs = 'langs_%s' % str(pretrained_decoder['langs'])
    pretrained_langs = 'langs_%s' % str(['de','el','en','es','fr','it','pt'])
    pretrained_target_token_selection = 'target_token_selection_%s' % str(pretrained_decoder['target_token_selection'])

    num_layers = 'num_layers_%s' % str(num_layers)
    pretrained_embedding = 'embedding_%s' % str(embedding_dim)
    split_save_path_dir = save_path_dir.split('/')    
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    seed_path = split_save_path_dir[seed_index]
    if pretrained_dataset == 'chapman':
        leads = "leads_['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']"
        #path = os.path.join(basepath,decoder_type,num_layers,pretrained_embedding,pretrained_supervision,pretrained_langs,pretrained_target_token_selection,pretrained_replacement_prob,seed_path,pretrained_dataset,leads,'training_fraction_1.00/finetuned_weight')
    elif pretrained_dataset in ['ptbxl','mimic','brazil']:
        leads = ''
    path = os.path.join(basepath,decoder_type,num_layers,pretrained_embedding,pretrained_supervision,pretrained_langs,pretrained_target_token_selection,pretrained_replacement_prob,seed_path,pretrained_dataset,leads,'training_fraction_1.00/finetuned_weight')
    params = torch.load(path)
    
    """ Modify MARGE Decoder Names To Match During IC """
    if pretraining_task in ['MARGE']:
        new_params = dict()
        for name,param in params.items():
            new_name = '.'.join(name.split('.')[1:])
            new_params[new_name] = param
        params = new_params
    
    print('Decoder Params Loaded!')        
    return params

def load_initial_model(phases,trial_to_load,save_path_dir,saved_weights,held_out_lr,nencoders,embedding_dim,dataset_name,encoder_type='cnn_small',decoder_type='lstm',num_layers=1,auxiliary_loss=False,attn=False,input_type='single',
                       goal='IC',pretrained_embeddings=False,freeze_embeddings=False,pretrained_encoder=False,freeze_encoder=False,pretrained_decoder=False,freeze_decoder=False,classification='',dest_lang_list=['en'],
                       target_token_selection='uniform',replacement_prob=0.15,labelled_fraction=1):
        
    """ Load Encoder """
    dropout_type = 'drop1d' #options: | 'drop1d' | 'drop2d'
    p1,p2,p3 = 0.1,0.1,0.1 #initial dropout probabilities #0.2, 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    print(device)

    noutputs = obtain_noutputs(classification,dataset_name,goal)
    if goal in ['IC','VQA','Supervised']:
        if encoder_type == 'cnn_small':
            encoder = image_encoder_network(dropout_type,p1,p2,p3,noutputs,embedding_dim,device,auxiliary_loss=auxiliary_loss,input_type=input_type)
        elif encoder_type == 'resnet50':
            encoder = resnet1d.resnet50(embedding_dim=embedding_dim,num_classes=noutputs)
        elif encoder_type == 'resnet34':
            encoder = resnet1d.resnet34(embedding_dim=embedding_dim,num_classes=noutputs)
        elif encoder_type == 'resnet18':
            encoder = resnet1d.resnet18(embedding_dim=embedding_dim,num_classes=noutputs)

        if pretrained_encoder is not False:
            """ Load Pretrained Encoder Parameters """
            params = load_encoder_params(encoder_type,embedding_dim,pretrained_encoder,save_path_dir)
            replace_params(encoder,params)
            if freeze_encoder == True:
                freeze_params(encoder)

    """ Load Token 2 ID Mapping Dict """
    """ Initialize token2id_dict to Add Tokens From Different Language(s) """
    token2id_dict = dict() 
    if 'chapman' in dataset_name or 'ptbxl' in dataset_name or 'brazil' in dataset_name:
        token2id_dict, df = obtain_token_mapping(dataset_name,dest_lang_list,labelled_fraction)
#        for dest_lang in dest_lang_list:
#            current_token2id_dict, nlp = obtain_token_mapping(dataset_name,dest_lang)
#            token2id_dict[dest_lang] = current_token2id_dict
    else: #mimic
        token2id_dict, df = obtain_token_mapping(dataset_name,dest_lang_list,labelled_fraction)
#        current_token2id_dict = obtain_token_mapping(dataset_name,dest_lang)
#        token2id_dict = current_token2id_dict
    
    """ Generate Word Embeddings """
    if 'train1' in phases:
        if pretrained_embeddings is not False:
            id2embedding_dict = load_embeddings(decoder_type,num_layers,embedding_dim,pretrained_embeddings,token2id_dict,save_path_dir,device)#,nlp)
            if freeze_embeddings == True:
                id2embedding_dict = {key:value.requires_grad_(False) for key,value in id2embedding_dict.items()}
                print('Embeddings Frozen!')
            
        elif pretrained_embeddings == False:
            """ Initialize All Vectors Randomly """
            id2embedding_dict = dict()
            for dest_lang in dest_lang_list:
                current_id2embedding_dict = dict(zip(token2id_dict[dest_lang].values(),[torch.rand(embedding_dim,requires_grad=True,device=device) for _ in range(len(token2id_dict[dest_lang]))]))
                id2embedding_dict[dest_lang] = current_id2embedding_dict
    else:
        id2embedding_dict = torch.load(os.path.join(save_path_dir,'id2embedding_dict')) #must load these because they ARE used during inference

    items_to_print = dict()
    labels = []
    for dest_lang,current_id2embedding_dict in id2embedding_dict.items():
        label = '%s Tokens' % dest_lang.upper()
        labels.append(label)
        items_to_print[label] = [len(current_id2embedding_dict)]
    print(tabulate(items_to_print,labels))

    #print('Number of Tokens: %i' % len(id2embedding_dict))
    if goal == 'VQA':
        """ Load Text Encoder """
        text_encoder = text_encoder_network(embedding_dim)
        """ Load Decoder """
        vqa_decoder = vqa_decoder_network(embedding_dim,attn=attn)
        """ Load Combined Model """
        model = combined_vqa_network(encoder,text_encoder,vqa_decoder)
    elif goal == 'IC':
        """ Load Decoder """
        
        """ Think About Duplicating Decoders Here if You Want (Maybe in the Form of a Module Dict Mapping from dest_lang to decoder) """
        vocab = {dest_lang:list(current_token2id_dict.keys()) for dest_lang,current_token2id_dict in token2id_dict.items()} #determines the number of outputs
        
        if pretrained_decoder is not False:
            pretraining_task = pretrained_decoder['supervision']
            if pretraining_task in ['MARGE']:
                decoder = MARGE_decoder_network(embedding_dim,embedding_dim,vocab,dest_lang_list,num_layers,head_depth=0,pretrained_decoder=pretrained_decoder)
            else:
                decoder = text_decoder_network(embedding_dim,vocab,attn=attn,goal=goal,decoder_type=decoder_type,num_layers=num_layers,dest_lang_list=dest_lang_list)
        else:
            decoder = text_decoder_network(embedding_dim,vocab,attn=attn,goal=goal,decoder_type=decoder_type,num_layers=num_layers,dest_lang_list=dest_lang_list)
        
        if pretrained_decoder is not False:
            """ Load Pretrained Decoder Parameters """
            params = load_decoder_params(decoder_type,num_layers,embedding_dim,pretrained_decoder,save_path_dir,pretraining_task)
            replace_params(decoder,params)
            print('Decoder Params Loaded!')
            if freeze_decoder == True:
                freeze_params(decoder)
        
        """ Load Combined Model """
        model = combined_image_captioning_network(encoder,decoder)
    elif goal == 'Supervised':
        model = encoder #image encoder 
    elif goal in ['Text_Supervised','Language_Change_Detection','Language_Detection','MLM','ELECTRA']:
        #vocab = list(token2id_dict.keys())
        vocab = {dest_lang:list(current_token2id_dict.keys()) for dest_lang,current_token2id_dict in token2id_dict.items()} #determines the number of outputs
        model = text_decoder_network(embedding_dim,vocab,attn=attn,noutputs=noutputs,goal=goal,decoder_type=decoder_type,num_layers=num_layers,dest_lang_list=dest_lang_list,target_token_selection=target_token_selection,replacement_prob=replacement_prob)
    elif goal in ['MARGE']:
        vocab = {dest_lang:list(current_token2id_dict.keys()) for dest_lang,current_token2id_dict in token2id_dict.items()} #determines the number of outputs        
        encoder = text_encoder_network(embedding_dim,text_encoder_type=decoder_type,num_layers=num_layers)
        decoder = MARGE_decoder_network(embedding_dim,embedding_dim,vocab,dest_lang_list,num_layers,pretrained_decoder=pretrained_decoder)
        model = MARGE(encoder,decoder,dataset_name)

    if 'test' in phases and len(phases) == 1 or 'val' in phases and len(phases) == 1:
        model.load_state_dict(torch.load(os.path.join(save_path_dir,saved_weights)))
        print('Finetuned Weights Loaded!')
    
    """ Send Model to Device """
    model.to(device)
    
    """ Load Optimizer """
    params = list(model.parameters())
    embeddings = []
    for dest_lang in dest_lang_list:
        current_embeddings = list(id2embedding_dict[dest_lang].values())
        embeddings = embeddings + current_embeddings
    all_params = params + embeddings #should work b/c list + list
    optimizer = optim.Adam(all_params,lr=held_out_lr,weight_decay=0)
    
   # optimizer = optim.AdamW(list(model.parameters()),lr=held_out_lr,weight_decay=0.001) #shouldn't load this again - will lose running average of gradients and so forth

    return model, token2id_dict, id2embedding_dict, optimizer, device, df

def obtain_noutputs(classification,dataset_name,goal):
    if 'physionet2020' in dataset_name:
        noutputs = 9 #int(classification.split('-')[0])
    elif 'ptbxl' in dataset_name:
        noutputs = 5 #5      #12 #71
    elif 'chapman' in dataset_name:
        if goal == 'VQA':
            noutputs = 101
        elif goal == 'Supervised':
            noutputs = 4
    elif 'mimic' in dataset_name:
        if goal == 'Text_Supervised':
            noutputs = 5
    elif 'brazil' in dataset_name:
        noutputs = 6
    elif classification == '2-way':
        noutputs = 1
    else:
        noutputs = int(classification.split('-')[0])
    return noutputs

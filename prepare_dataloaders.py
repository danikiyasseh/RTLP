#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:22:28 2020

@author: Dani Kiyasseh
"""

from torch.utils.data import DataLoader
from prepare_dataset import my_dataset
from torch.nn.utils.rnn import pad_sequence
import torch
""" Functions in this script:
    1) load_initial_data
"""
#%%

class Load_Data(object):
    
    def __init__(self,dataset_name,goal='IC'):
        self.goal = goal
        self.dataset_name = dataset_name
    
    def load_initial_data(self,basepath_to_data,phases,df,fraction,inferences,batch_size,modality,acquired_indices,acquired_labels,modalities,token2id_dict,input_perturbed=False,perturbation='Gaussian',band_types=['frequency','time'],nbands=[1,1],band_width=[0.1,0.1],leads='ii',labelled_fraction=1,unlabelled_fraction=1,downstream_task='contrastive',class_pair='',trial='CMC',output_type='single',input_type='single',dest_lang_list='en',nviews=1):    
        """ Control augmentation at beginning of training here """ 
        resize = False
        affine = False
        rotation = False
        color = False    
        perform_cutout = False
        operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
        shuffles = {'train1':True,
                    'train2':False,
                    'val': False,
                    'test': False}
        
        fractions = {'fraction': fraction,
                     'labelled_fraction': labelled_fraction,
                     'unlabelled_fraction': unlabelled_fraction}
        
        acquired_items = {'acquired_indices': acquired_indices,
                          'acquired_labels': acquired_labels}
        
        dataset = {phase:my_dataset(basepath_to_data,self.dataset_name,phase,df,inference,fractions,acquired_items,token2id_dict,modalities=modalities,task=downstream_task,input_perturbed=input_perturbed,perturbation=perturbation,band_types=band_types,nbands=nbands,band_width=band_width,leads=leads,class_pair=class_pair,trial=trial,nviews=nviews,output_type=output_type,input_type=input_type,dest_lang_list=dest_lang_list,goal=self.goal) for phase,inference in zip(phases,inferences)}                                        
                    
        dataloader = {phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False,collate_fn=self.pad_collate) for phase in phases}
        return dataloader,operations

    def pad_collate(self,batch):
        """ This Pads The Text Data to Have Equal Lengths """
        frame, text, class_label, modality, dataset, true_index = zip(*batch)
        """ Process Every Entry """
        frame = torch.stack([f for f in frame])
        #sentence_lens = [len(sentence)-1 for sentence in text] #-1 to account for '/START' token
        #text_padded = pad_sequence(text, batch_first=True, padding_value=0)
        text_padded = dict()  
        sentence_lens = dict()
        language_dict = dict()
        
        document_level_text_padded = dict()
        document_level_sentence_lens = dict()
        #torch.save(text,'text')
        
        if self.goal == 'MARGE':
            if 'ptbxl' in self.dataset_name:
                target_lang = 'de'
            elif 'brazil' in self.dataset_name:
                target_lang = 'pt'

            all_other_lang_text_indices = []
            all_target_lang_text_indices = []
            
            all_other_languages = []
            all_target_languages = []
            
            all_other_sentence_lens = []
            all_target_sentence_lens = []
            
            dest_lang_list = text[0].keys()       
            """ Iterate Over Instances To Form Mini-Batch """
            for instance in text:
                other_lang_text_indices = []
                other_languages = []
                other_sentence_len = 0
                """ Iterate Over Languages Within Same Instance """
                for dest_lang in dest_lang_list:
                    if dest_lang != target_lang:
                        dest_lang_text_indices = instance[dest_lang]
                        other_lang_text_indices.extend(dest_lang_text_indices)
                        """ Obtain Sentence Length """
                        sentence_length = len(dest_lang_text_indices)-1 
                        other_sentence_len += sentence_length
                        """ Obtain Language Label """
                        languages = [dest_lang for _ in range(len(dest_lang_text_indices))]
                        other_languages.extend(languages)
                    else:
                        target_lang_text_indices = instance[dest_lang]
                        """ Obtain Language Label """
                        target_languages = [dest_lang for _ in range(len(target_lang_text_indices))]
                        """ Obtain Sentence Len """
                        target_sentence_len = len(target_lang_text_indices)-1

                """ Flatten the Other Lang Text Indices """
                other_lang_text_indices = torch.tensor(other_lang_text_indices)
                all_other_lang_text_indices.append(other_lang_text_indices)
                all_other_languages.append(other_languages)
                all_other_sentence_lens.append(other_sentence_len)
                other_max_length = max(all_other_sentence_lens)
                """ Build Target Lang Text Indices """
                all_target_lang_text_indices.append(target_lang_text_indices)
                all_target_languages.append(target_languages)
                all_target_sentence_lens.append(target_sentence_len)
                target_max_length = max(all_target_sentence_lens)

            """ Package Data Into Dictionaries """
            modified_langs = ['other',target_lang]
            modified_indices = [all_other_lang_text_indices,all_target_lang_text_indices]
            modified_languages = [all_other_languages,all_target_languages]
            modified_sentence_lens = [all_other_sentence_lens,all_target_sentence_lens]
            for modified_lang,modified_index,modified_language,modified_sentence_len in zip(modified_langs,modified_indices,modified_languages,modified_sentence_lens):
                """ Store Text Indices """
                current_text_padded = pad_sequence(modified_index, batch_first=True, padding_value=0)
                text_padded[modified_lang] = current_text_padded           
                """ Store Languages """
                max_length = other_max_length if modified_lang == 'other' else target_max_length
                """ For Some This Reason, This Changed modified_language In Place (No Assignment Necessary) """
                #NOTE ********* lang[0] is used as a filler, will not be considered later on 
                [lang.extend([lang[0]]*(max_length - sentence_len)) for lang,sentence_len in zip(modified_language,modified_sentence_len)]
                language_dict[modified_lang] = modified_language
                """ Store Sentence Lengths """
                sentence_lens[modified_lang] = modified_sentence_len


            """ This is Needed for Docuument Level (Language-Level) Representations """
            for dest_lang in text[0].keys():
                dest_text = [t[dest_lang] for t in text]
                current_sentence_lens = [len(sentence[dest_lang])-1 for sentence in text] #-1 to account for '/START' token
                document_level_sentence_lens[dest_lang] = current_sentence_lens
                current_text_padded = pad_sequence(dest_text, batch_first=True, padding_value=0)
                document_level_text_padded[dest_lang] = current_text_padded
        else:
            for dest_lang in text[0].keys():
                dest_text = [t[dest_lang] for t in text]
                current_sentence_lens = [len(sentence[dest_lang])-1 for sentence in text] #-1 to account for '/START' token
                sentence_lens[dest_lang] = current_sentence_lens
                current_text_padded = pad_sequence(dest_text, batch_first=True, padding_value=0)
                text_padded[dest_lang] = current_text_padded
                languages = [dest_lang for _ in range(len(dest_text))]
                language_dict[dest_lang] = languages
        #text_padded = {dest_lang : pad_sequence(dest_text, batch_first=True, padding_value=0) for dest_lang_to_tensor_dict in text for dest_text in dest_lang_to_tensor_dict.values()} #text is a tuple ({'en':tensor,'de':tensor}, ...)
        class_label = torch.stack([p for p in class_label])
        #modality = [mod for mod in modality]
        dataset = [d for d in dataset]
        #true_index = [i for i in true_index]
        
        return frame, text_padded, sentence_lens, class_label, language_dict, document_level_text_padded, document_level_sentence_lens




    
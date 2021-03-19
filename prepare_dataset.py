#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:16:38 2020

@author: Dani Kiyasseh
"""
import torch
from torch.utils.data import Dataset
from operator import itemgetter
import os
import pickle
import numpy as np
import random
import scipy.signal as signal
from tqdm import tqdm
from load_ptbxl_data import modify_df
import load_chapman_ecg_v2
from prepare_brazil_ecg import load_annotations_and_modify_labels
import wfdb
#from prepare_brazil_ecg import load_annotations_and_modify_labels

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
from spacy_pipeline_101 import obtain_token_mapping, load_ptbxl_df, load_nlp, modify_text
from translate_brazil_text import load_brazil_df
import load_mimic_df
import pandas as pd
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

#%%

class my_dataset(Dataset):
    """ Takes Arrays and Phase, and Returns Sample 
        i.e. use for BIDMC and PhysioNet Datasets 
    """
    
    def __init__(self,basepath_to_data,dataset_name,phase,df,inference,fractions,acquired_items,token2id_dict,modalities=['ecg','ppg'],task='contrastive',input_perturbed=False,perturbation='Gaussian',band_types=['frequency','time'],
                 nbands=[1,1],band_width=[0.1,0.1],leads='ii',heads='single',cl_scenario=None,class_pair='',trial='CMC',input_type='single',output_type='single',dest_lang_list=['en'],nviews=1,goal='IC'):
        """ This Accounts for 'train1' and 'train2' Phases """
        if 'train' in phase:
            phase = 'train'
        elif 'val' in phase:
            phase = 'val'

        #nlp_de = spacy.load('de_core_news_sm')
        #nlp_en = spacy.load('en_core_web_sm')
        #self.nlp_de = nlp_de
        #self.nlp_en = nlp_en

#        """ Obtain Vocab (Training Set Always) """
#        if 'chapman' in dataset_name:
#            self.nlp = spacy.load('en_core_web_sm')
#        elif 'ptbxl' in dataset_name:
#            self.nlp = load_nlp(dest_lang)
            #self.nlp = spacy.load('de_core_news_md') #'sm' | 'md'
        self.nlp = {dest_lang: load_nlp(dest_lang) for dest_lang in dest_lang_list}
        self.dest_lang_list = dest_lang_list
            #token2id_dict, self.nlp = obtain_token_mapping(dataset_name)
        #elif 'mimic' in dataset_name:
        #    token2id_dict = obtain_token_mapping(dataset_name)
        
        #print(len(token2id_dict),np.max(list(token2id_dict.values())))
        #token2id_dict, self.nlp = obtain_token_mapping(dataset_name)
        
        #vocab = list(token2id_dict.keys())
        vocab = {dest_lang: list(current_token2id_dict.keys()) for dest_lang,current_token2id_dict in token2id_dict.items()}
        self.token2id_dict = token2id_dict
        self.vocab = vocab
        
        """ Load PTBXL DF with Modified Report Text (Includes Start and End) """
#        if 'ptbxl' in dataset_name:
#            df = load_ptbxl_df(dest_lang_list)
#            #""" Obtain Language Specific Column and Add Special Tokens """
#            #df = add_start_end_symbols(df,dest_lang)
#        elif 'chapman' in dataset_name:
#            df, max_values, min_values = load_chapman_ecg_v2.modify_df()
#        elif 'mimic' in dataset_name:
#            df = load_mimic_df.modify_df()
#        elif 'brazil' in dataset_name:
#            df = load_brazil_df(dest_lang_list)
            
        self.df = df
        
        #if 'contrastive' in task:
        #    task = 'contrastive'
        self.task = task #continual_buffer, etc. 
        self.cl_scenario = cl_scenario
        self.basepath_to_data = basepath_to_data
        self.dataset_name = dataset_name

        fraction = fractions['fraction'] #needs to be a list when dealing with 'query' or inference = True for CL scenario
        labelled_fraction = fractions['labelled_fraction']
        self.fraction = fraction
        self.labelled_fraction = labelled_fraction
        """ Combine Modalities into 1 Array """
        self.modalities = modalities
        self.heads = heads
        self.acquired_items = acquired_items
        self.leads = leads
        self.class_pair = class_pair
        self.nviews = nviews
        self.trial = trial
        self.input_perturbed = input_perturbed
        self.perturbation = perturbation
        self.phase = phase
        self.output_type = output_type
        self.input_type = input_type
        self.band_types,self.nbands,self.band_width = band_types, nbands, band_width
        self.goal = goal
#        self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks
        
        if phase == 'train':
            inputs, outputs, pids = self.retrieve_multi_task_train_data()
        else:
            inputs, outputs, pids = self.retrieve_multi_task_val_data(phase)
        
        keep_indices = list(np.arange(inputs.shape[0])) #filler
        modality_array = list(np.arange(inputs.shape[0])) #filler
        dataset_list = ['All' for _ in range(len(keep_indices))] #filler
            
        #dataset_list = [self.name for _ in range(len(keep_indices))] #filler
        self.dataset_name = dataset_name
        self.dataset_list = dataset_list
        self.input_array = inputs
        self.label_array = outputs
        self.pids = pids
        self.modality_array = modality_array
        self.remaining_indices = keep_indices
        
        self.input_perturbed = input_perturbed 

    def load_raw_inputs_and_outputs(self,dataset_name,leads='i'):
        """ Load Arrays Based on dataset_name """
        #basepath = '/home/scro3517/Desktop'
        basepath = self.basepath_to_data
        
        if dataset_name == 'bidmc':
            path = os.path.join(basepath,'BIDMC v1')
            extension = 'heartpy_'
        elif dataset_name == 'physionet':
            path = os.path.join(basepath,'PhysioNet v2')
            extension = 'heartpy_'
        elif dataset_name == 'mimic':
            shrink_factor = str(0.1)
            path = os.path.join(basepath,'MIMIC3_WFDB','frame-level',shrink_factor)
            extension = 'heartpy_'
        elif dataset_name == 'cipa':
            lead = ['II','aVR']
            path = os.path.join(basepath,'cipa-ecg-validation-study-1.0.0','leads_%s' % lead)
            extension = ''
        elif dataset_name == 'cardiology':
            leads = 'all' #flexibility to change later 
            path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data',self.task,'%s_classes' % leads)
            extension = ''
        elif dataset_name == 'physionet2017':
            path = os.path.join(basepath,'PhysioNet 2017','patient_data',self.task)
            extension = ''
        elif dataset_name == 'tetanus':
            path = '/media/scro3517/TertiaryHDD/new_tetanus_data/patient_data'
            extension = ''
        elif dataset_name == 'ptb':
            leads = [leads]
            path = os.path.join(basepath,'ptb-diagnostic-ecg-database-1.0.0','patient_data','leads_%s' % leads)
            extension = ''  
        elif dataset_name == 'fetal':
            abdomen = leads #'Abdomen_1'
            path = os.path.join(basepath,'non-invasive-fetal-ecg-arrhythmia-database-1.0.0','patient_data',abdomen)
            extension = ''
        elif dataset_name == 'physionet2016':
            path = os.path.join(basepath,'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0')
            extension = ''
        elif dataset_name == 'physionet2020':
            #basepath = '/mnt/SecondaryHDD'
            leads_name = leads
            path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data',self.task,'leads_%s' % leads_name)
            extension = ''
        elif dataset_name == 'chapman':
            #basepath = '/mnt/SecondaryHDD'
            """ OLD Loading Style """
            #leads = leads
            #path = os.path.join(basepath,'chapman_ecg',self.task,'leads_%s' % leads)
            #extension = ''
            """ New Loading Style """
            output_type = self.output_type        
            input_type = self.input_type            
            
        elif dataset_name == 'chapman_pvc':
            #basepath = '/mnt/SecondaryHDD'
            leads = leads
            path = os.path.join(basepath,'PVCVTECGData',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'ptbxl':
            #basepath = '/mnt/SecondaryHDD'
            #leads = leads
            #code_of_interest = 'diagnostic_class' #diagnostic_class' #options: 'rhythm' | 'all' | 'diagnostic_class' #tells you the classification formulation 
            #output_type = 'single'
            ##seconds = str(5) #options: 2.5 | 5 seconds
            #path = os.path.join(basepath,'PTB-XL','patient_data','contrastive_ss','leads_%s' % leads,'classes_%s' % code_of_interest,'%s_output' % output_type)#,'%s_seconds' % seconds)
            #extension = ''
            output_type = self.output_type        
            input_type = self.input_type
        
        if self.cl_scenario == 'Class-IL':
            dataset_name = dataset_name + '_' + 'mutually_exclusive_classes'
        
#        if dataset_name not in ['ptbxl','chapman']:
#            """ Dict Containing Actual Frames """
#            with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
#                input_array = pickle.load(f)
#            """ Dict Containing Actual Labels """
#            with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
#                output_array = pickle.load(g)
#            """ Dict Containing Patient Numbers """
#            with open(os.path.join(path,'pid_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as h:
#                pid_array = pickle.load(h) #needed for CPPC (ours)
        if dataset_name in ['ptbxl']:
            #input_array, output_array, pid_array = self.merge_ptbxl_data(path,dataset_name)
            loadpath = os.path.join(basepath,'PTB-XL','patient_data','%s_input' % input_type,'%s_output' % output_type)
            with open(os.path.join(loadpath,'phase_to_paths.pkl'),'rb') as f:
                phase_to_paths = pickle.load(f)
            with open(os.path.join(loadpath,'phase_to_leads.pkl'),'rb') as f:
                phase_to_leads = pickle.load(f) #not used right now
            self.phase_to_leads = phase_to_leads
            with open(os.path.join(loadpath,'phase_to_segment_number.pkl'),'rb') as f:
                phase_to_segment_number = pickle.load(f)
            self.phase_to_segment_number = phase_to_segment_number
        elif dataset_name in ['chapman']:
            loadpath = os.path.join(basepath,'chapman_ecg','patient_data',self.goal,'%s_input' % input_type,'%s_output' % output_type)
            with open(os.path.join(loadpath,'phase_to_paths.pkl'),'rb') as f:
                phase_to_paths = pickle.load(f)
            with open(os.path.join(loadpath,'phase_to_leads.pkl'),'rb') as f:
                phase_to_leads = pickle.load(f) #not used right now
            self.phase_to_leads = phase_to_leads
            with open(os.path.join(loadpath,'phase_to_segment_number.pkl'),'rb') as f:
                phase_to_segment_number = pickle.load(f)
            self.phase_to_segment_number = phase_to_segment_number
            with open(os.path.join(loadpath,'phase_to_questions.pkl'),'rb') as f:
                phase_to_questions = pickle.load(f)
            self.phase_to_questions = phase_to_questions
        elif dataset_name in ['brazil']:
            loadpath = os.path.join(basepath,'Brazil_ECG','patient_data')
            with open(os.path.join(loadpath,'year_to_paths.pkl'),'rb') as f:
                year_to_paths = pickle.load(f)
            
        if dataset_name in ['ptbxl']:
            input_array, output_array, pid_array = self.load_ptbxl_data(phase_to_paths,output_type=output_type)
        elif dataset_name in ['chapman']:
            input_array, output_array, pid_array = self.load_chapman_data(phase_to_paths,output_type=output_type)
        elif dataset_name in ['mimic']:
            input_array, output_array, pid_array = self.load_mimic_data()
        elif dataset_name in ['brazil']:
            input_array, output_array, pid_array = self.load_brazil_data(year_to_paths)
            
        return input_array,output_array,pid_array

    def load_brazil_data(self,year_to_paths):
        phase = self.phase
        
        input_array, output_array, pid_array = dict(), dict(), dict()
        input_array['ecg'], output_array['ecg'], pid_array['ecg'] = dict(), dict(), dict()
        input_array['ecg'][1], output_array['ecg'][1], pid_array['ecg'][1] = dict(), dict(), dict()
        input_array['ecg'][1][phase], output_array['ecg'][1][phase], pid_array['ecg'][1][phase] = dict(), dict(), dict()
        
        basepath = '/mnt/SecondaryHDD/Brazil_ECG'
        annot_df = load_annotations_and_modify_labels(basepath)
        
        #text_df = self.df
        
        all_paths = []
        all_ids = []
        all_class_labels = []
        for year,paths in year_to_paths.items():
            """ Obtain Paths """
            current_phase_paths = paths[phase]
            all_paths.extend(current_phase_paths)
            """ Obtain IDs """
            ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0].split('TNMG')[1]),current_phase_paths))
            all_ids.extend(ids) #these are exam_ids
            """ Obtain Labels """
            #annot_df.id_exam.isin(ids).single_label
            class_labels = [annot_df[annot_df['id_exam'] == id_entry].single_label.item() for id_entry in ids]
            all_class_labels.extend(class_labels)
            
        """ Inputs = Paths """
        input_array['ecg'][1][phase][''] = all_paths
        pid_array['ecg'][1][phase][''] = all_ids
        output_array['ecg'][1][phase][''] = all_class_labels

        return input_array, output_array, pid_array

    def load_mimic_data(self,output_type='single'):
        phase = self.phase
        
        input_array, output_array, pid_array = dict(), dict(), dict()
        input_array['ecg'], output_array['ecg'], pid_array['ecg'] = dict(), dict(), dict()
        input_array['ecg'][1], output_array['ecg'][1], pid_array['ecg'][1] = dict(), dict(), dict()
        input_array['ecg'][1][phase], output_array['ecg'][1][phase], pid_array['ecg'][1][phase] = dict(), dict(), dict()

        df = self.df
                
        text_categories = df[df.Phase == phase]['TextCategory']
        pids = df[df.Phase == phase]['SUBJECT_ID']
        
        input_array['ecg'][1][phase][''] = text_categories
        output_array['ecg'][1][phase][''] = text_categories
        pid_array['ecg'][1][phase][''] = pids 
        
        return input_array, output_array, pid_array

    def load_chapman_data(self,phase_to_paths,output_type='single'):
        """ Load Chapman Data """
        phase = self.phase
        
        input_array, output_array, pid_array = dict(), dict(), dict()
        input_array['ecg'], output_array['ecg'], pid_array['ecg'] = dict(), dict(), dict()
        input_array['ecg'][1], output_array['ecg'][1], pid_array['ecg'][1] = dict(), dict(), dict()
        input_array['ecg'][1][phase], output_array['ecg'][1][phase], pid_array['ecg'][1][phase] = dict(), dict(), dict()
        
        #df = self.df

        paths = phase_to_paths[phase] #this is the most essential part
        """ Inputs = Paths """
        input_array['ecg'][1][phase][''] = paths
        """ Obtain IDs """
        filenames = list(map(lambda path:path.split('/')[-1].split('.csv')[0],paths))

        """ Obtain Report Text For Each Entry """
        questions = self.phase_to_questions[phase]
        output_array['ecg'][1][phase][''] = questions

        #""" Obtain Labels """
        #if output_type == 'single':
        #    labels = [df['Rhythm'][df['FileName'] == filename].iloc[0] for filename in filenames]
        
        #answers = [df[df['FileName'] == filename][question].iloc[0] for question,filename in zip(questions,filenames)]
        
        """ Repurposing the PID Array to Contains Class Labels """
        pid_array['ecg'][1][phase][''] = filenames

        return input_array, output_array, pid_array

    def load_ptbxl_data(self,phase_to_paths,output_type='single'):
        """ Load PTB-XL Data """
        phase = self.phase
        
        input_array, output_array, pid_array = dict(), dict(), dict()
        input_array['ecg'], output_array['ecg'], pid_array['ecg'] = dict(), dict(), dict()
        input_array['ecg'][1], output_array['ecg'][1], pid_array['ecg'][1] = dict(), dict(), dict()
        input_array['ecg'][1][phase], output_array['ecg'][1][phase], pid_array['ecg'][1][phase] = dict(), dict(), dict()
        
        df = self.df

        paths = phase_to_paths[phase] #this is the most essential part
        """ Inputs = Paths """
        input_array['ecg'][1][phase][''] = paths
        """ Obtain IDs """
        ecg_ids = list(map(lambda path:int(path.split('/')[-1].split('_')[0]),paths))

        """ Obtain Labels """
        if output_type == 'single':
            labels = [df[df.index == id_entry].superdiagnostic_label.iloc[0] for id_entry in ecg_ids]
        elif output_type == 'multi':
            labels = [df[df.index == id_entry].iloc[0][-5:].tolist() for id_entry in ecg_ids]

        """ Repurposing the PID Array to Contains Class Labels """
        pid_array['ecg'][1][phase][''] = labels
        
        """ Obtain Report Text For Each Entry """
        text = [df[df.index == id_entry].report.iloc[0] for id_entry in ecg_ids]
        #text = [df[df.index == id_entry].new_report.iloc[0] for id_entry in ecg_ids]
        output_array['ecg'][1][phase][''] = text

        return input_array, output_array, pid_array

    def offset_outputs(self,dataset_name,outputs,t=0): #t tells you which class pair you are on now (used rarely and only for MTL)
        """ Offset Label Position in case of Single Head """
        dataset_and_offset = self.acquired_items['noutputs']
        if self.heads == 'single':
            """ Changed March 17th 2020 """
            offset = dataset_and_offset[dataset_name] #self.dataset_name
            """ End """
            if dataset_name == 'physionet2020': #multilabel situation 
                """ Option 1 - Expansion """
                #noutputs = outputs.shape[1] * 12 #9 classes and 12 leads
                #expanded_outputs = np.zeros((outputs.shape[0],noutputs))
                #expanded_outputs[:,offset:offset+9] = outputs
                #outputs = expanded_outputs
                """ Option 2 - No Expansion """
                outputs = outputs 
            else: 
                if dataset_name == 'cardiology' and self.task == 'multi_task_learning':
                    outputs = outputs + 2*t
                elif dataset_name == 'chapman' and self.task == 'multi_task_learning':
                    outputs = outputs
                else:
                    outputs = outputs + offset #output represents actual labels
                #print(offset)
        return outputs

    def retrieve_buffered_data(self,buffer_indices_dict,fraction,labelled_fraction):
        input_buffer = []
        output_buffer = []
        task_indices_buffer = []
        dataset_buffer = []
        #print(fraction_list)
        #for fraction,(task_name,indices) in zip(fraction_list[:-1],buffer_indices_dict.items()):
        for task_name,indices in buffer_indices_dict.items():
            #name = '-'.join((task,modality,leads,str(fraction))) #dataset, modality, fraction, leads
            dataset = task_name.split('-')[0]
            fraction = float(task_name.split('-')[2])
            leads = task_name.split('-')[3]
            if self.cl_scenario == 'Class-IL':
                self.class_pair = '-'.join(task_name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
            elif self.cl_scenario == 'Time-IL':
                self.class_pair = task_name.split('-')[-1] 
            elif self.cl_scenario == 'Task-IL' and 'chapman' in dataset: #chapman ecg as task in Task-IL setting
                self.class_pair = task_name.split('-')[-1] 
            input_array,output_array = self.load_raw_inputs_and_outputs(dataset,leads)
            input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=dataset)
            """ Offset Applied to Each Dataset """
            if self.heads == 'single':#'continual_buffer':
                output_array = self.offset_outputs(dataset,output_array)
                #offset = self.dataset_and_offset[dataset]
                #output_array = output_array + offset
            current_input_buffer,current_output_buffer = input_array[indices,:], output_array[indices]
            input_buffer.append(current_input_buffer)
            output_buffer.append(current_output_buffer)
            task_indices_buffer.append(indices) #will go 1-10K, 1-10K, etc. not cumulative indices
            dataset_buffer.append([task_name for _ in range(len(indices))])
        #print(input_buffer)
        input_buffer = np.concatenate(input_buffer,axis=0)
        output_buffer = np.concatenate(output_buffer,axis=0)
        task_indices_buffer = np.concatenate(task_indices_buffer,axis=0)
        dataset_buffer = np.concatenate(dataset_buffer,axis=0)
        return input_buffer,output_buffer,task_indices_buffer,dataset_buffer
                
    def expand_labelled_data_with_buffer(self,input_array,output_array,buffer_indices_dict,fraction,labelled_fraction):
        """ function arguments are raw inputs and outputs """
        input_buffer,output_buffer,task_indices_buffer,dataset_buffer = self.retrieve_buffered_data(buffer_indices_dict,fraction,labelled_fraction)
        if self.cl_scenario == 'Class-IL':
            self.class_pair = '-'.join(self.name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
        elif self.cl_scenario == 'Time-IL':
            self.class_pair = self.name.split('-')[-1]
        input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=self.dataset_name)
        dataset_list = [self.name for _ in range(input_array.shape[0])]
        #print(max(output_array))
        """ Offset Applied to Current Dataset """
        if self.heads == 'single':#'continual_buffer':
            output_array = self.offset_outputs(self.dataset_name,output_array)
            #offset = self.dataset_and_offset[self.dataset_name]
            #print('Offset')
            #print(offset)
            #output_array = output_array + offset
        print(input_array.shape,input_buffer.shape)
        input_array = np.concatenate((input_array,input_buffer),0)
        output_array = np.concatenate((output_array,output_buffer),0)
        dataset_list = np.concatenate((dataset_list,dataset_buffer),0)
        #print(input_array.shape)
        #print(max(output_array),max(output_buffer))
        return input_array,output_array,dataset_list
    
    def retrieve_val_data(self,input_array,output_array,pid_array,phase,fraction,labelled_fraction=1,dataset_name=''):#,modalities=['ecg','ppg']):
        frame_array = []
        label_array = []
        pids = []
        #if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or (self.cl_scenario == 'Task-IL' and self.dataset_name == 'chapman'):        
        if dataset_name in ['chapman','ptbxl','mimic','brazil']:
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase][self.class_pair]
                modality_output = output_array[modality][fraction][phase][self.class_pair]
                modality_pids = pid_array[modality][fraction][phase][self.class_pair]
                frame_array.append(modality_input)
                label_array.append(modality_output)
                pids.append(modality_pids)
        else:
            """ Obtain Modality-Combined Unlabelled Dataset """
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase]
                modality_output = output_array[modality][fraction][phase]
                modality_pids = pid_array[modality][fraction][phase]
                frame_array.append(modality_input)
                label_array.append(modality_output) 
                pids.append(modality_pids)
        
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)
        pids = np.concatenate(pids)
        
        inputs,outputs,pids,_ = self.shrink_data(inputs,outputs,pids,labelled_fraction)
        
        return inputs,outputs,pids          
    
    def shrink_data(self,inputs,outputs,pids,fraction,modality_array=None):
        nframes_to_sample = int(inputs.shape[0]*fraction)
        random.seed(0) #to make sure we always obtain SAME shrunken dataset for reproducibility 
        indices = random.sample(list(np.arange(inputs.shape[0])),nframes_to_sample)
        inputs = np.array(list(itemgetter(*indices)(inputs)))
        outputs = np.array(list(itemgetter(*indices)(outputs)))
        pids = np.array(list(itemgetter(*indices)(pids)))
        if modality_array is not None:
            modality_array = np.array(list(itemgetter(*indices)(modality_array)))
        return inputs,outputs,pids,modality_array
    
    def remove_acquired_data(self,inputs,outputs,modality_array,acquired_indices):
        keep_indices = list(set(list(np.arange(inputs.shape[0]))) - set(acquired_indices))
        inputs = np.array(list(itemgetter(*keep_indices)(inputs)))
        outputs = np.array(list(itemgetter(*keep_indices)(outputs)))
        modality_array = np.array(list(itemgetter(*keep_indices)(modality_array)))
        return inputs,outputs,modality_array,keep_indices
    
    def retrieve_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []
        modality_array = []
        
        """ Obtain Modality-Combined Unlabelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase]['unlabelled']
            modality_output = output_array[modality][fraction][phase]['unlabelled']
            modality_name = [modality for _ in range(modality_input.shape[0])]
            frame_array.append(modality_input)
            label_array.append(modality_output)
            modality_array.append(modality_name)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)   
        modality_array = np.concatenate(modality_array)         
        
        inputs,outputs,modality_array = self.shrink_data(inputs,outputs,unlabelled_fraction,modality_array)
        
        return inputs,outputs,modality_array
        
    ### This is function you want for MC Dropout Phase ###
    def retrieve_modified_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(input_array,output_array,fraction,unlabelled_fraction)
        inputs,outputs,modality_array,keep_indices = self.remove_acquired_data(inputs,outputs,modality_array,acquired_indices)
        return inputs,outputs,modality_array,keep_indices

    def retrieve_labelled_data(self,input_array,output_array,pid_array,fraction,labelled_fraction,dataset_name=''):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []
        pids = []

        if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or dataset_name == 'chapman':
            header = self.class_pair
        elif self.cl_scenario == 'Task-IL' and dataset_name == 'chapman':
            header = self.class_pair
        else:
            header = 'labelled'
        
        if 'ptbxl' in self.dataset_name or 'mimic' in self.dataset_name or 'brazil' in self.dataset_name:
            header = ''
        
        """ Obtain Modality-Combined Labelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase][header]
            modality_output = output_array[modality][fraction][phase][header]
            modality_pids = pid_array[modality][fraction][phase][header]
            frame_array.append(modality_input)
            label_array.append(modality_output)
            pids.append(modality_pids)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)
        pids = np.concatenate(pids)
        
        inputs,outputs,pids,_ = self.shrink_data(inputs,outputs,pids,labelled_fraction)

        return inputs,outputs,pids

    def acquire_unlabelled_samples(self,inputs,outputs,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(inputs,outputs,fraction,unlabelled_fraction)
        if len(acquired_indices) > 1:
            inputs = np.array(list(itemgetter(*acquired_indices)(inputs)))
            outputs = np.array(list(itemgetter(*acquired_indices)(outputs)))
            modality_array = np.array(list(itemgetter(*acquired_indices)(modality_array)))
        elif len(acquired_indices) == 1:
            """ Dimensions Need to be Adressed to allow for Concatenation """
            inputs = np.expand_dims(np.array(inputs[acquired_indices[0],:]),1)
            outputs = np.expand_dims(np.array(outputs[acquired_indices[0]]),1)
            modality_array = np.expand_dims(np.array(modality_array[acquired_indices[0]]),1)
        return inputs,outputs,modality_array

#    def retrieve_multi_task_train_data_ptbxl(self):
#        """ Load All Required Tasks for Multi-Task Training Setting """
#        all_class_pairs = self.class_pair
#        all_modalities = self.modalities
#        input_array = []
#        output_array = []
#        pids = []
#        dataset_name_list = ['ptbxl']*12
#        fraction_list = [1]*12
#        leads_list =  [['I'], ['II'], ['III'], ['AVR'], ['AVL'], ['AVF'], ['V1'], ['V2'],['V3'], ['V4'], ['V5'], ['V6']]
#        
#        #for t,(dataset_name,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
#        for t,(dataset_name,fraction,leads) in enumerate(zip(dataset_name_list,fraction_list,leads_list)):#,all_class_pairs)): #should be an iterable list
#
#            #leads_all = [['I'], ['II'], ['III'], ['AVR'], ['AVL']]
#            #current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads_all[t])
#            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
#            self.class_pair = '5' #class_pair
#            self.modalities = ['ecg'] #all_modalities[t] #list(map(lambda x:x[0],all_modalities))
#            #print(self.labelled_fraction)
#            current_input, current_output, pid = self.retrieve_labelled_data(current_input,current_output,pid,fraction,self.labelled_fraction,dataset_name=dataset_name)
#            #current_output = self.offset_outputs(dataset_name,current_output,t)
#            #print(current_output.shape)
#            input_array.append(current_input)
#            output_array.append(current_output)
#            pids.append(pid)
#        input_array = np.concatenate(input_array,axis=0)
#        output_array = np.concatenate(output_array,axis=0)
#        pids = np.concatenate(pids,axis=0)
#        print('Output Dimension: %s' % str(output_array.shape))
#        print('Maximum Output Index: %i' % np.max(output_array))
#        return input_array,output_array,pids

#    def retrieve_multi_task_val_data_ptbxl(self,phase):
#        """ Load All Required Tasks for Multi-Task Validation/Testing Setting """
#        all_class_pairs = self.class_pair
#        all_modalities = self.modalities
#        input_array = []
#        output_array = []
#        pids = []
#        dataset_name_list = ['ptbxl']*12
#        fraction_list = [1]*12
#        leads_list =  [['I'], ['II'], ['III'], ['AVR'], ['AVL'], ['AVF'], ['V1'], ['V2'],['V3'], ['V4'], ['V5'], ['V6']]
#        
#        #for t,(dataset_name,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
#        for t,(dataset_name,fraction,leads) in enumerate(zip(dataset_name_list,fraction_list,leads_list)):#,all_class_pairs)): #should be an iterable list
#        #for t,(dataset_name,modalities,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,all_modalities,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
#            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
#            self.class_pair = '5' #class_pair
#            self.modalities = ['ecg']# modalities
#            current_input, current_output, pid = self.retrieve_val_data(current_input,current_output,pid,phase,fraction,dataset_name=dataset_name)#,labelled_fraction=1)
#            #current_output = self.offset_outputs(dataset_name,current_output,t)
#            input_array.append(current_input)
#            output_array.append(current_output)
#            pids.append(pid)
#        input_array = np.concatenate(input_array,axis=0)
#        output_array = np.concatenate(output_array,axis=0)
#        pids = np.concatenate(pids,axis=0)
#        return input_array,output_array,pids

    def retrieve_multi_task_train_data(self):
        """ Load All Required Tasks for Multi-Task Training Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        pids = []
        for t,(dataset_name,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            #leads_all = [['I'], ['II'], ['III'], ['AVR'], ['AVL']]
            #current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads_all[t])
            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = all_modalities[t] #list(map(lambda x:x[0],all_modalities))
            #print(self.labelled_fraction)
            current_input, current_output, pid = self.retrieve_labelled_data(current_input,current_output,pid,fraction,self.labelled_fraction,dataset_name=dataset_name)
            #current_output = self.offset_outputs(dataset_name,current_output,t)
            #print(current_output.shape)
            input_array.append(current_input)
            output_array.append(current_output)
            pids.append(pid)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        pids = np.concatenate(pids,axis=0)
        print('Output Dimension: %s' % str(output_array.shape))
        #print('Maximum Output Index: %i' % np.max(output_array))
        return input_array,output_array,pids

    def retrieve_multi_task_val_data(self,phase):
        """ Load All Required Tasks for Multi-Task Validation/Testing Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        pids = []
        for t,(dataset_name,modalities,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,all_modalities,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = modalities
            current_input, current_output, pid = self.retrieve_val_data(current_input,current_output,pid,phase,fraction,dataset_name=dataset_name)#,labelled_fraction=1)
            #current_output = self.offset_outputs(dataset_name,current_output,t)
            input_array.append(current_input)
            output_array.append(current_output)
            pids.append(pid)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        pids = np.concatenate(pids,axis=0)
        
        return input_array,output_array,pids

    ### This is function you want for training ###
    def expand_labelled_data(self,input_array,output_array,pid_array,fraction,labelled_fraction,unlabelled_fraction,acquired_indices,acquired_labels):
        inputs,outputs,pids = self.retrieve_labelled_data(input_array,output_array,pid_array,fraction,labelled_fraction,self.dataset_name)
        #print(self.remaining_indices)
        #print('Acquired Indices!')
        #print(acquired_indices)
        """ If indices have been acquired, then use them. Otherwise, do not """
        #if isinstance(acquired_indices,list):
        #    condition = len(acquired_indices) > 0
        #elif isinstance(acquired_indices,dict):
        #    condition = len(acquired_indices) > 1
        """ Changed March 5, 2020 """
        #if len(acquired_indices) > 0:
        if len(acquired_indices) > 0:
            acquired_inputs,acquired_outputs,acquired_modalities = self.acquire_unlabelled_samples(input_array,output_array,fraction,unlabelled_fraction,acquired_indices)
            inputs = np.concatenate((inputs,acquired_inputs),0)
            #print(acquired_labels)
            #""" Note - Acquired Labels from Network Predictions are Used, Not Ground Truth """
            #acquired_labels = np.fromiter(acquired_labels.values(),dtype=float)
            acquired_labels = np.array(list(acquired_labels.values()))
            acquired_labels = acquired_labels.reshape((-1,))
            ##""" For cold_gt trials, run this line """
            ##acquired_labels = np.array(list(acquired_labels.values()))
            ##acquired_labels = acquired_labels.reshape((-1,))
            ##print(outputs.shape,acquired_labels.shape)
            ##""" End GT Labels """
            
            #print(acquired_labels)
            outputs = np.concatenate((outputs,acquired_labels),0) 
        return inputs,outputs,pids
    
    def obtain_sampling_rate(self):
        if self.dataset_name == 'cardiology':
            fs = 200
        elif self.dataset_name == 'physionet2017':
            fs = 300
        elif self.dataset_name == 'physionet2020':
            fs = 500
        elif self.dataset_name == 'chapman':
            fs = 250
        return fs
    
    def spec_augment(self,frame,band_types,nbands,band_width):
        """
        Args:
            frame (numpy array): original input frame 
            band_types (list of str): list of strings indicating whether to mask frequency or spectral components
            nbands (list of ints): list of integers indicating number of bands to mask in each dimension
            band_width (list of ints): indicating the fraction of bins to mask 
        Returns:
            frame (numpy array): masked version of the frame
        """
        """ Obtain Sampling Rate for Dataset """
        fs = self.obtain_sampling_rate()
        """ STFT of Frame """
        fbands,tbands,stft = signal.stft(frame,fs,nperseg=fs//2,padded=True)
        """ Number of Frequency Bins, Number of Time Bins """
        n_fbands, n_tbands = len(fbands), len(tbands)
        """ Iterate Over Dimension e.g. Frequency or Time """
        for band_type,nband,width in zip(band_types,nbands,band_width):
            """ Iterate Over the Number of Bands in this particular Dimension """
            for i in range(nband):
                if band_type == 'frequency':
                    """ Low Frequency Components Are of Interest When Masking """
                    n_fbands_of_interest = 20 #i.e. take first 20 frequency bins
                    """ Width of Mask in Bins """
                    mask_width = int(n_fbands_of_interest*width)
                    """ Start Bin for Masking """
                    mask_start = np.random.randint(0,n_fbands_of_interest - mask_width)
                    """ Perform Masking """
                    stft[mask_start : mask_start + mask_width,:] = 0+0j 
                elif band_type == 'time':
                    """ Width of Mask in Bins """
                    mask_width = int(n_tbands*width)
                    """ Start Bin for Masking """
                    mask_start = np.random.randint(0,n_tbands - mask_width)
                    """ Perform Masking """
                    stft[:,mask_start : mask_start + mask_width] = 0+0j
        """ Perform ISTFT to Produce Frame """
        _,frame = signal.istft(stft,fs,nperseg=fs//2)
        """ Only Obtain First 2500 Samples (In The Event Reconstructed Frame is Slightly Longer) """
        frame = frame[:2500]
        return frame
    
    def obtain_perturbed_frame(self,frame):
        """ Apply Sequence of Perturbations to Frame 
        Args:
            frame (numpy array): frame containing ECG data
        Outputs
            frame (numpy array): perturbed frame based
        """
        if self.input_perturbed:
            """ Apply Perturbations Sequentially """
            for perturbation in self.perturbation: #self.perturbation is a list of perturbations
                if 'Gaussian' in perturbation:
                    """ Additive Gaussian Noise """
                    mult_factor = 1
                    if self.dataset_name in ['ptb','ptbxl','physionet2020']:
                        variance_factor = 0.01*mult_factor
                    elif self.dataset_name in ['cardiology','chapman']:
                        variance_factor = 10*mult_factor
                    elif self.dataset_name in ['physionet','physionet2017']:
                        variance_factor = 100*mult_factor 
                    gauss_noise = np.random.normal(0,variance_factor,size=(2500))
                    frame = frame + gauss_noise
                elif 'FlipAlongY' in perturbation:
                    """ Horizontal Flip of Frame """
                    frame = np.flip(frame)
                elif 'FlipAlongX' in perturbation:
                    """ Vertical Flip of Frame """
                    frame = -frame
                elif 'SpecAugment' in perturbation:
                    frame = self.spec_augment(frame,self.band_types,self.nbands,self.band_width)
        return frame

    def normalize_frame(self,frame):
        if self.dataset_name not in ['cardiology','physionet2017','physionet2016']:# or self.dataset_name != 'physionet2017':# or self.dataset_name != 'cipa':
            if isinstance(frame,np.ndarray):
                frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8)
            elif isinstance(frame,torch.Tensor):
                frame = (frame - torch.min(frame))/(torch.max(frame) - torch.min(frame) + 1e-8)
        return frame

    def bandpass_filter(self,frame):
        if 'ptbxl' in self.dataset_name:
            b, a = signal.butter(2, [0.002,0.05], btype='bandpass') #0.002 = 0.5 Hz / 250 Hz (which is Nyquist Frequency since Fs = 500 Hz) 
            frame = signal.filtfilt(b,a,frame).copy()
        return frame

    def convert_text_to_indices(self,text,dest_lang):
        if 'ptbxl' in self.dataset_name:
            #nlp = self.nlp#_de
            """ Split According to Whitespace """ 
            report_text = text.split() #step 1 for each instance during training/inference
            """ Convert List to String """
            report_text = ' '.join(report_text) #step 2 for each instance during training/inference
            
            """ Apply Tokenizer to Text """
            doc = self.nlp[dest_lang](report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
            """ Remove Punctuation and Return Lemmatized Word """
            tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
        elif 'chapman' in self.dataset_name:
            #nlp = self.nlp#_en
            """ Split According to Whitespace """ 
            report_text = text.split() #step 1 for each instance during training/inference
            """ Convert List to String """
            report_text = ' '.join(report_text) #step 2 for each instance during training/inference
            """ Apply Tokenizer to Text """
            doc = self.nlp[dest_lang](report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
            """ Remove Punctuation and Return Lemmatized Word """
            tokens = [token.text for token in doc]# if token.is_punct == False] #step 4 for each instance during training/inference
        elif 'mimic' in self.dataset_name:
            #nlp_en = self.nlp#_en
            """ Split According to Whitespace """ 
            report_text = text.split() #step 1 for each instance during training/inference
            """ Convert List to String """
            report_text = ' '.join(report_text) #step 2 for each instance during training/inference
            """ Apply Tokenizer to Text """
            doc = tokenizer(report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
            """ Remove Punctuation and Return Lemmatized Word """
            tokens = [token.lower_ for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
        elif 'brazil' in self.dataset_name:
            """ Split According to Whitespace """ 
            report_text = text.split() #step 1 for each instance during training/inference
            """ Convert List to String """
            report_text = ' '.join(report_text) #step 2 for each instance during training/inference
            
            """ Apply Tokenizer to Text """
            doc = self.nlp[dest_lang](report_text) #step 3 for each instance during training/inference (you could technically just replace with tokenizer)
            """ Remove Punctuation and Return Lemmatized Word """
            #tokens = [token.text.lower() for token in doc if token.is_punct == False] #step 4 for each instance during training/inference
            tokens = [token.text.lower() for token in doc if token.is_alpha == True or token.text in ['/START','/END']] #step 4 for each instance during training/inference
        
        """ Make Sure Tokens are in Vocab """
        tokens = [token if token in self.vocab[dest_lang] else 'OOD' for token in tokens]
        """ Convert Entries to String """
        tokens = list(map(lambda x: str(x),tokens)) #step 5 for each instance during training/inference
        """ Map Tokens to Index """
        word_indices = list(itemgetter(*tokens)(self.token2id_dict[dest_lang]))
        return word_indices

    def __getitem__(self,index):
        true_index = self.remaining_indices[index] #this should represent indices in original unlabelled set
        input_frame = self.input_array[index]
        label = self.label_array[index] #text
        """ NOTE: we redefined PID when loading ptbxl above (for convenience) """
        class_label = self.pids[index] 
        modality = self.modality_array[index]
        dataset = self.dataset_list[index] #task name
        #print(input_frame,label,pid)
        if 'ptbxl' in self.dataset_name:
            ecg_id = int(input_frame.split('/')[-1].split('_')[0])
            data = wfdb.rdsamp(input_frame)
            """ Obtain List of Leads to Choose From """
            leads_list = self.phase_to_leads[self.phase]
            current_lead = leads_list[index]
            """ Obtain Segment Number """
            segment_numbers = self.phase_to_segment_number[self.phase]
            segment_number = segment_numbers[index]
            """ Obtain ECG Input Data  """
            input_frame = data[0]
            """ Obtain Lead of Interest """
            if self.input_type == 'single': #single lead in input 
                lead_index = np.where(np.in1d(data[1]['sig_name'],[current_lead]))[0]
                input_frame = input_frame[:,lead_index]
                """ Obtain Segment """
                input_frame = input_frame[2500*segment_number : 2500*(segment_number+1)]
            elif self.input_type == 'multi': #multiple leads in input
                input_frame = input_frame
                """ Obtain Segment """
                input_frame = input_frame[2500*segment_number : 2500*(segment_number+1),:] #Lx12
            """ Filter Frame """
            input_frame = self.bandpass_filter(input_frame.transpose()) #12xL
            """ Remove First Second (Due to Filtering Process) """
            input_frame = input_frame[:,500:] #12Xl
            #""" Remove Baseline Wander """
            #frames = self.filter_low_pass(frames) #12xL
            """ Downsample to 2500 """
            input_frame = signal.resample(input_frame.transpose(),2500).transpose() #12x2500
            """ Normalize Signal """
            #input_frame = np.array([(frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8) for frame in input_frame])
            """ Standardize Signal """
            input_frame = np.array([(frame - np.mean(frame))/(np.std(frame) + 1e-8) for frame in input_frame])
            
            if self.goal in ['Supervised','Text_Supervised']:
                """ Obtain Labels """
                if self.output_type == 'single':
                    class_label = self.df[self.df.index == ecg_id].superdiagnostic_label.iloc[0]
                elif self.output_type == 'multi':
                    class_label = self.df[self.df.index == ecg_id].iloc[0][-5:].tolist()
            
            text_indices = dict()
            for dest_lang in self.dest_lang_list:
                """ Obtain Language Specific Text """
                column_of_interest = '%s_report' % dest_lang #column without start and end tokens
                text = self.df[self.df.index == ecg_id][column_of_interest].iloc[0]
                """ Include Start and End Tokens """
                text_modifier = modify_text(dest_lang,'ptbxl') 
                text = text_modifier.add_start_end_symbols_to_single_report(text)
                #text = add_start_end_symbols_to_single_report(text)
                #print(text)
                """ Obtain Indices By Using Correct Token2Id Mapping """
                current_text_indices = self.convert_text_to_indices(text,dest_lang)
                current_text_indices = torch.tensor(current_text_indices)            
                text_indices[dest_lang] = current_text_indices
            
        elif 'chapman' in self.dataset_name:
            filename = input_frame.split('/')[-1].split('.csv')[0]
            input_frame = pd.read_csv(input_frame,header=None) #5000x12
            """ Obtain List of Leads to Choose From """
            leads_list = self.phase_to_leads[self.phase]
            current_lead = leads_list[index]
            """ Obtain Segment Number """
            segment_numbers = self.phase_to_segment_number[self.phase]
            segment_number = segment_numbers[index]

            if self.goal == 'VQA':
                """ Load Answer to Question """
                question = self.phase_to_questions[self.phase][index]    
                answer = self.df[self.df['FileName'] == filename][question].iloc[0]
                """ Answer for Regression """
                #class_label = answer # \in R
                """ Answer for Classification """
                bins = np.linspace(-1,1,101) #granularity of the bins 
                class_label = np.digitize(answer,bins)-1 #-1 is needed to get indices from 0 to C-1   
            elif self.goal == 'Supervised':
                rhythm_to_class_mapping = dict(zip(['AFIB', 'GSVT', 'SB', 'SR'],range(4)))
                rhythm = self.df[self.df['FileName'] == filename]['Rhythm'].iloc[0]
                rhythm = rhythm_to_class_mapping[rhythm]
                class_label = rhythm
            """ Obtain Lead of Interest """
#            if self.input_type == 'single': #single lead in input 
#                lead_index = np.where(np.in1d(data[1]['sig_name'],[current_lead]))[0]
#                input_frame = input_frame[:,lead_index]
#                """ Obtain Segment """
#                input_frame = input_frame.iloc[2500*segment_number : 2500*(segment_number+1)]
            if self.input_type == 'multi': #multiple leads in input
                """ Obtain Segment """
                input_frame = input_frame.iloc[2500*segment_number:2500*(segment_number+1),:].to_numpy() #Lx12
            #""" Filter Frame """
            #input_frame = self.bandpass_filter(input_frame.transpose()) #12xL
            #""" Remove First Second (Due to Filtering Process) """
            #input_frame = input_frame[:,500:] #12Xl
            #""" Remove Baseline Wander """
            #frames = self.filter_low_pass(frames) #12xL
            """ Downsample to 2500 """
            input_frame = signal.resample(input_frame,2500).transpose() #12x2500
            """ Normalize Signal """
            input_frame = np.array([(frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8) for frame in input_frame])
            """ Standardize Signal """
            #input_frame = np.array([(frame - np.mean(frame))/(np.std(frame) + 1e-8) for frame in input_frame])
        elif 'mimic' in self.dataset_name:
            subject_id = self.pids[index] 
            label = self.df[self.df.SUBJECT_ID == subject_id].TEXT.iloc[0] #actual text
            if self.goal == 'Text_Supervised':
                class_label = self.df[self.df.SUBJECT_ID == subject_id].TextCategory.iloc[0] #text category 
        elif 'brazil' in self.dataset_name:
            """ input_frame is going to be a path directory """
            data = wfdb.rdsamp(input_frame)
            """ Obtain ECG Input Data  """
            frames = data[0]
            """ Remove Baseline Wander """
            frames = self.bandpass_filter(frames)
            """ Downsample to 2500 """
            input_frame = signal.resample(frames,2500).transpose()  #12xL
            """ Obtain Current Exam ID """
            id_exam = self.pids[index]
            
            text_indices = dict()
            for dest_lang in self.dest_lang_list: #should only be 'pt' for now
                """ Obtain Language Specific Text """
                column_of_interest = '%s_report' % dest_lang #column without start and end tokens                
                text = self.df[self.df.ID_EXAME == id_exam][column_of_interest].iloc[0] #text based df
                """ Include Start and End Tokens """
                text_modifier = modify_text(dest_lang,'brazil')
                text = text_modifier.add_start_end_symbols_to_single_report(text)

                """ Obtain Indices By Using Correct Token2Id Mapping """
                current_text_indices = self.convert_text_to_indices(text,dest_lang)
                current_text_indices = torch.tensor(current_text_indices)            
                text_indices[dest_lang] = current_text_indices
        
#        """ Keep """
#        """ Convert Report Text to Indices """
#        text_indices = self.convert_text_to_indices(label,dest_lang)
#        text_indices = torch.tensor(text_indices)
#        """ End """

        """ Convert Frame to Tensor """
        class_label = torch.tensor(class_label,dtype=torch.float) #long for multi-class, #float for binary
                
        #print(index,frame,text_indices)
        #print(all((frame==0).tolist()))
        
        if 'mimic' not in self.dataset_name:
            frame = torch.tensor(input_frame,dtype=torch.float)
            if len(frame.shape) == 1: #only do so for single lead situations 
                frame = frame.unsqueeze(0) #(1,5000)
        else:
            frame = torch.zeros(1,1)
        #frame_views = frame.unsqueeze(2) #to show that there is only 1 view (1x2500x1) #this is needed for OUR network #remove for resnet1d
        #else:
        #    frame_views = frame
        return frame, text_indices, class_label, modality, dataset, true_index
        
    def __len__(self):
        return len(self.input_array)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:14:18 2020

@author: Dani Kiyasseh
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import einsum
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from operator import itemgetter
import numpy as np
import random
from einops import rearrange, repeat
from spacy_pipeline_101 import modify_text
text_modifier = modify_text('en','filler') #random language plays no role
start_dict, end_dict = text_modifier.obtain_start_and_end_dict()
#%%
""" Functions in this script:
    1) full_network
"""
#%%

class image_encoder_network(nn.Module):
    
    def __init__(self,dropout_type,p1,p2,p3,noutputs,embedding_dim,device,auxiliary_loss=False,input_type='single'):
        super(image_encoder_network,self).__init__()

        if input_type == 'single':
            c1 = 1 # 1 b/c single time-series
        elif input_type == 'multi':
            c1 = 12
        c2 = 32 #4 #32 for physionet2020 and chapman
        c3 = 64 #16 #64 for physionet2020 and chapman
        c4 = 128 #32 #128 for physionet2020 and chapman
        k=7 
        s=3 
        
        self.conv1 = nn.Conv1d(c1,c2,k,s)
        self.batchnorm1 = nn.BatchNorm1d(c2)
        self.conv2 = nn.Conv1d(c2,c3,k,s)
        self.batchnorm2 = nn.BatchNorm1d(c3)
        self.conv3 = nn.Conv1d(c3,c4,k,s)
        self.batchnorm3 = nn.BatchNorm1d(c4)     
        #self.linear1 = nn.Linear(c4*10,embedding_dim)#*4) #*4
        #self.linear2 = nn.Linear(c4*10,noutputs)
        self.linear1 = nn.Linear(c4,embedding_dim)#*4) #*4
        self.linear2 = nn.Linear(embedding_dim,noutputs)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
        self.dropout2 = nn.Dropout(p=p2) #0.2
        self.dropout3 = nn.Dropout(p=p3)
        self.maxpool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.noutputs = noutputs
        self.device = device 
        
    def forward(self,x):
        """ Traditional Neural Network Forward Pass 
        Args:
            x (Tensor): batch input (BxS)
            v_dict (dict): dictionary of patients and v vectors
            patients (list): list of patients in batch (B)
        Outputs:
            outputs (Tensor): batch posterior distributions (BxC)
        """
        x = self.dropout1(self.maxpool(self.relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool(self.relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool(self.relu(self.batchnorm3(self.conv3(x))))) #128x128x10 #BxC4xL 
        
        x = x.permute(0,2,1) #128x10x128
        annot_vectors = self.linear1(x).permute(0,2,1) #128x300x10
        h = self.avgpool(annot_vectors).squeeze() #128x300
        #x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        #h = self.linear1(x)
        #annot_vectors = x
        class_predictions = self.linear2(h)
        return h, annot_vectors, class_predictions

#%%
class text_encoder_network(nn.Module):
    
    def __init__(self,embedding_dim,text_encoder_type='lstm',num_layers=4):
        super(text_encoder_network,self).__init__()
        
        input_features = embedding_dim #word embedding dims #keep as same dimension as hidden to allow for attention calculation
        hidden_features = embedding_dim

        if text_encoder_type == 'lstm':
            """ LSTM NETWORK """
            self.num_layers = num_layers
            network = nn.LSTM(input_features,hidden_features,num_layers,batch_first=True)
        elif text_encoder_type == 'transformer':
            """ TRANSFORMER NETWORK """
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_features, nhead=5)
            network = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.network = network
        self.text_encoder_type = text_encoder_type
    
    def forward(self,text_indices,sentence_lens,languages,token2id_dict,id2embedding_dict,phase,device):
        """ Forward Pass of the Text Encoder Across Tokens From All Language Documents 
        Args:
            text_indices (torch.Tensor): list of text indices
        Returns:
            representations (torch.Tensor): token representations for all documents 
        """

        """ Pass Through LSTM Time-Step """
        text_indices = text_indices.tolist() # BxS
        """ Convert Padded Sequences to Embeddings - Option 1""" 
        #embeddings = torch.stack([torch.stack(itemgetter(*text_index)(itemgetter(*language_index)(id2embedding_dict))) for text_index,language_index in zip(text_indices,languages)],0) #BxSxL
        """ Option 2 """
        embeddings = torch.stack([torch.stack([id2embedding_dict[l][t] for t,l in zip(text_index,language_index)]) for text_index,language_index in zip(text_indices,languages)],0) #BxSxL
        
        #ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
        if self.text_encoder_type == 'lstm':
            batch_size = len(text_indices)
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L

            #for sentence_length in range(1,ntokens): #ntokens is the same for all sentences in the same batch due to padding introduced with dataloader
            #    token_indices = text_indices[:,sentence_length-1].tolist() # BxL
            #    embeddings = torch.stack([id2embedding_dict[token_index] for token_index in token_indices]) # BxL
            #    embeddings = embeddings.unsqueeze(1) # Bx1xL
            #    """ Forward Pass and Obtain Probabilities Over Next Word """
            representations, (h, c) = self.network(embeddings,(h,c))

            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach() 
        elif self.text_encoder_type == 'transformer':
#            """ Obtain Text Indices in List Form """
#            text_indices = text_indices.tolist() # BxS
#            """ Convert Padded Sequences to Embeddings """ 
#            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict)) for text_index in text_indices],0) #BxSxL
            embeddings = embeddings.permute(1,0,2)
            # no need for mask since bidirectional encoder is fine (looking ahead is fine with encoder)
            """ CREATE MASK TO AVOID CONSIDERING PADDED REGIONS """
            key_padding_mask = torch.zeros(embeddings.shape[1],embeddings.shape[0],device=device).type(torch.bool) #BxS
            for row,sentence_len in zip(range(key_padding_mask.shape[0]),sentence_lens):
                key_padding_mask[row,sentence_len:] = True #ignore these indices marked with True

            representations = self.network(embeddings,src_key_padding_mask=key_padding_mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
        
        return representations

    def get_embeddings(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,phase,device):
        """ Forward Pass of the Text Encoder Across Tokens From All Language Documents 
        Args:
            text_indices (torch.Tensor): list of text indices
        Returns:
            representations (torch.Tensor): token representations for all documents 
        """

        """ Pass Through LSTM Time-Step """
        text_indices = text_indices.tolist() # BxS
        """ Convert Padded Sequences to Embeddings - Option 1""" 
        embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict)) for text_index in text_indices],0) #BxSxL
        
        if self.text_encoder_type == 'lstm':
            batch_size = len(text_indices)
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            representations, (h, c) = self.network(embeddings,(h,c))
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach() 
        elif self.text_encoder_type == 'transformer':
            embeddings = embeddings.permute(1,0,2)
            # no need for mask since bidirectional encoder is fine (looking ahead is fine with encoder)
            """ CREATE MASK TO AVOID CONSIDERING PADDING REGIONS """
            key_padding_mask = torch.zeros(embeddings.shape[1],embeddings.shape[0],device=device).type(torch.bool) #BxS
            for row,sentence_len in zip(range(key_padding_mask.shape[0]),sentence_lens):
                key_padding_mask[row,sentence_len:] = True #ignore these indices marked with True
                
            representations = self.network(embeddings,src_key_padding_mask=key_padding_mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
        
        """ Document Summary Representation is First Token Representation """
        representations = representations[0,:,:] # B x L 
        
        return representations

#%%
def exists(x):
    return x is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 4, causal = True, dropout = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        _, n, _, h, device = *x.shape, self.heads, x.device # x is B x S x L ?
        qkv = self.to_qkv(x)
        #print(x.shape,qkv.shape)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', h = h, qkv = 3)
        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        if exists(mask):
            #""" New """
            #mask = mask.type(torch.bool)
            #""" End """
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        if self.causal:
            causal_mask = torch.ones(n, n, device=device).triu_(1).bool()
            dots.masked_fill_(causal_mask, float('-inf'))
            del causal_mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 4, dropout = 0.):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, doc_similarities = None, mask = None, context_mask = None, target_lang = None, document_level_sentence_lens = None):
        b, n, _, h, device = *x.shape, self.heads, x.device

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        context_len = context.shape[1]  # B x S x L
        #context = rearrange(context, 'b m n d -> b (m n) d') #n is S in my world 
        #context_mask = rearrange(context_mask, 'b m n -> b (m n)') if exists(context_mask) else None        
        

        if doc_similarities is not None:
#            doc_similarities = repeat(doc_similarities, 'b m -> b m n', n=context_len) # adding a token-level dimension to document level tensor
#            doc_similarities = rearrange(doc_similarities, 'b m n -> b (m n)') #m documents with n tokens each
#            doc_similarities = doc_similarities[:, None, None, :] * self.beta
            """ Only Needed to Account for My Formatting """
            old_doc_similarities = doc_similarities
            
            """ These Values Will be Changed to Account for Document-Specific Format I Have """
            doc_similarities = repeat(doc_similarities, 'b m -> b m n', n=context_len) # adding a token-level dimension to document level tensor
            doc_similarities = rearrange(doc_similarities, 'b m n -> b (m n)') #m documents with n tokens each
            doc_similarities = doc_similarities[:,:context_len]
            doc_similarities = doc_similarities[:, None, None, :] * self.beta

            """ Extend Doc Similarities to Tokens Based on Sentence Lens """
            #target_lang = 'de'
            other_langs = set(list(document_level_sentence_lens.keys())) - set([target_lang])
            for i in range(context.shape[0]): #iterate over instances in batch
                start = 0
                for l,dest_lang in enumerate(other_langs):
                    sentence_len = document_level_sentence_lens[dest_lang][i]
                    end = start + sentence_len
                    doc_similarities[i,0,0,start:end] = old_doc_similarities[i,l].repeat(sentence_len)
                    start = end         
        #print(context.shape)
        kv = self.to_kv(context)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', h = h, kv = 2)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = dots + doc_similarities if doc_similarities is not None else dots 

        if any(map(exists, (mask, context_mask))):
            if not exists(mask):
                mask = torch.full((b, n), True, dtype=torch.bool, device=device)

            if not exists(context_mask):
                context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=device)

            cross_mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~cross_mask, float('-inf'))
            del cross_mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class MARGE_decoder_network(nn.Module):

    def __init__(self, input_features, hidden_features, vocab, dest_lang_list, depth, head_depth = 4, heads = 4, ff_mult = 4, attn_dropout = 0., ff_dropout = 0., pretrained_decoder=False, MAX_LENGTH=45):
        super().__init__()
        dim = input_features
        self.decoder_head = nn.ModuleList([])
        self.decoder_tail = nn.ModuleList([])
        self.pretrained_decoder = pretrained_decoder
#        for _ in range(head_depth):
#            self.decoder_head.append(nn.ModuleList([
#                Residual(PreNorm(dim, SelfAttention(dim, causal = True, dropout = attn_dropout))),
#                Residual(PreNorm(dim, FeedForward(dim)))
#            ]))

        for _ in range(depth - head_depth):
            self.decoder_tail.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim, causal = True, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim))),
                Residual(PreNorm(dim, CrossAttention(dim, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mult = ff_mult)))
            ]))
                
        linear_dict = nn.ModuleDict()
        for dest_lang in dest_lang_list:
            linear = nn.Linear(hidden_features,len(vocab[dest_lang]))
            linear_dict[dest_lang] = linear
        self.linear_dict = linear_dict
        self.MAX_LENGTH = MAX_LENGTH

    def generate_mask_to_avoid_future_entries(self, output_sequence_length, device):#, sentence_lens):
        """ Mask To Avoid Looking at Future Entries """
        mask = (torch.triu(torch.ones(output_sequence_length, output_sequence_length,device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def obtain_outputs_without_teacher_forcing(self,text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device):
        """ List to Keep Track of Predicted Words """
        all_outputs = torch.zeros(len(text_indices),self.MAX_LENGTH,len(id2token_dict)) # B x S x L
        
        #if self.decoder_type == 'transformer':
        for b in range(len(text_indices)):
            instance_outputs = torch.zeros(1,self.MAX_LENGTH,len(id2token_dict))
            #instance_attn_coefs = torch.zeros(1,self.MAX_LENGTH+1,image_regions)
            """ Get Current Sentence """
            current_text_indices = text_indices[b]#.tolist()
            """ Get Current Word """
            current_text_index = current_text_indices[0] #should be index associated with '/START' for all sentences in batch
            current_token = id2token_dict[current_text_index]
            """ Current Embedding """
            embedding = id2embedding_dict[current_text_index]
            """ Change Dimension of Embedding """
            embedding = embedding.view(1,1,-1) #1x1xL
            """ Calculate Context Vector Based on Attention (Now Called Embedding Which is Already Concatenated) """
            instance_annot_vectors = annot_vectors[b,:,:].unsqueeze(0) #1 X 79 x 300
            """ Pass Through LSTM Time-Step """
            sentence_length = 0
            embedding_inputs = embedding
            end_token = end_dict[dest_lang].lower()
            while (current_token != end_token) and (sentence_length < self.MAX_LENGTH): 
                #print(current_token + ' ', end_token)
                #print(current_token == end_token)
                """ Forward Pass and Obtain Probabilities Over Next Word """
                decoder_input = embedding_inputs # 1 x 1 (growing) x L
                #encoder_output = instance_annot_vectors.permute(2,0,1) # 79x1xL
                encoder_output = instance_annot_vectors
                mask = self.generate_mask_to_avoid_future_entries(sentence_length+1,device)
                
                x = decoder_input
                context = encoder_output
                src_mask = mask 

                """ Perform Forward Pass Through Layers """
                for self_attn, self_ff, cross_attn, cross_ff in self.decoder_tail:
                    x = self_attn(x)#, mask = src_mask)
                    x = self_ff(x)
                    similarities = None #dont use similarities on downstream task 
                    #print(x.shape,context.shape)
                    x = cross_attn(x, context, similarities)#, mask = src_mask)#, context_mask = context_mask)
                    x = cross_ff(x) # B x S x L
                
                """ Obtain Predictions Over Words """
                curr_outputs = self.linear_dict[dest_lang](x) #BxSxV (V = Vocab)
                """ Once Sequence Grows, You Only Want Last Output """
                outputs = curr_outputs[:,-1,:] # 1 x V
                
                """ **** PREPARE FOR NEXT TIME-STEP **** """
                
                """ Obtain Predicted Word """
                current_text_index = torch.argmax(outputs).item()
                current_token = id2token_dict[current_text_index]  
                """ Obtain Embedding of Predicted Word """
                embedding = id2embedding_dict[current_text_index]
                embedding = embedding.view(1,1,-1) #1x1xL  
                """ Grow Your Embedding Inputs """
                embedding_inputs = torch.cat((embedding_inputs,embedding),1) # 1 x S (growing) x L
                """ Store Predictions Over Words """
                instance_outputs[0,sentence_length,:] = outputs
                #print(instance_outputs.shape)
                sentence_length += 1
            """ Store Predicted Words for This Sentence """
            all_outputs[b,:,:] = instance_outputs 
        
        return all_outputs 

                     #text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device,h0=None,c0=None,annot_vectors=None    
    def forward(self,text_indices,decoder_sentence_lens,token2id_dict,id2embedding_dict,target_lang,phase,sampling,device,h0=None,c0=None,annot_vectors=None,similarities=None,inference=False,document_level_sentence_lens=None):        
        id2token_dict = {value:key for key,value in token2id_dict.items()}  
        
        """ Define Encoder Output For Use in Multi-head Attention """
        encoder_output = annot_vectors.permute(1,0,2)
        context = encoder_output
        ntokens = len(text_indices[0])        
        """ Obtain Text Indices in List Form """
        text_indices = text_indices.tolist() # BxS
        """ Convert Padded Sequences to Embeddings """ 
        decoder_input = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict)) for text_index in text_indices],0) #BxSxL
        """ Mask for Decoder Input to Avoid Looking Forward """
        mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS

#        for self_attn, self_ff in self.decoder_head:
#            x = self_attn(x, mask = src_mask)
#            x = self_ff(x)
        if self.pretrained_decoder is not False: #implies we are in downstream task

            if 'train' in phase:
                context = context.permute(1,2,0) # B x S x L

                x = decoder_input
                src_mask = mask
        
                """ Perform Forward Pass Through Layers """
                for self_attn, self_ff, cross_attn, cross_ff in self.decoder_tail:
                    x = self_attn(x)#, mask = src_mask)
                    x = self_ff(x)
                    #similarities = None
                    #print(x.shape,context.shape)
                    x = cross_attn(x, context, similarities)#, mask = src_mask)#, context_mask = context_mask)
                    x = cross_ff(x) # B x S x L
                
                """ Obtain Predictions Over Words """
                all_outputs = self.linear_dict[target_lang](x) #BxSxV (V = Vocab)
                """ Do Not Consider Output When Input is /END """
                all_outputs = all_outputs[:,:-1,:] #drop last sequence output
            else:
                """ Inference Without Teacher Forcing """
                image_regions = encoder_output.shape[-1]
                annot_vectors = annot_vectors.permute(0,2,1)
                all_outputs = self.obtain_outputs_without_teacher_forcing(text_indices,decoder_sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,target_lang,image_regions,h0,c0,device)
        elif self.pretrained_decoder == False: #implies we are in the upstream pre-training task
            x = decoder_input
            src_mask = mask
    
            """ Perform Forward Pass Through Layers """
            for self_attn, self_ff, cross_attn, cross_ff in self.decoder_tail:
                x = self_attn(x)#, mask = src_mask)
                x = self_ff(x)
                x = cross_attn(x, context, similarities, target_lang = target_lang, document_level_sentence_lens = document_level_sentence_lens)#, mask = src_mask)#, context_mask = context_mask)
                x = cross_ff(x) # B x S x L
            
            """ Obtain Predictions Over Words """
            all_outputs = self.linear_dict[target_lang](x) #BxSxV (V = Vocab)
            """ Do Not Consider Output When Input is /END """
            all_outputs = all_outputs[:,:-1,:] #drop last sequence output
        
        return all_outputs
        
class MARGE(nn.Module):
    
    def __init__(self,encoder,decoder,dataset_name):
        super(MARGE,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dataset_name = dataset_name
    
    def obtain_cosine_similarity_matrix(self,all_document_representations,target_lang): #this is a dictionary with lang -> representations
        """ Obtain the Similarity Matrix Between Document in Target Language and Same Document in Other Languages 
        Args:
            all_document_representations (dict): contains document representations for each language
            target_lang (str): target language of interest 
        Returns:
            similarities (torch.Tensor): similarity matrix between document in target_lang and same document in other languages
        """
        
        """ Obtain Remaining Languages """
        dest_lang_list = all_document_representations.keys()
        other_langs = set(dest_lang_list) - set([target_lang]) # 
        """ Obtain Representations Associated with Remaining Languages """
        other_lang_representations = torch.stack([all_document_representations[lang] for lang in other_langs],dim=1) # B x M x L
        other_norms = torch.norm(other_lang_representations,dim=2).unsqueeze(-1).repeat(1,1,other_lang_representations.shape[-1]) # B x M x L
        """ Normalized Representations Associated with Remaining Languages """
        other_lang_representations_normed = torch.div(other_lang_representations,other_norms) # B x M x L
        
        """ Obtain Representations Associated with Target Language """
        target_lang_representations = all_document_representations[target_lang].unsqueeze(-1) # B x L x 1
        target_norms = torch.norm(target_lang_representations,dim=1).unsqueeze(1).repeat(1,target_lang_representations.shape[1],1) # B x L x 1 
        """ Normalize Representations Associated with Target Language """
        target_lang_representations_normed = torch.div(target_lang_representations,target_norms) # B x L x 1
        #print(other_lang_representations_normed.shape,target_lang_representations_normed.shape)
        """ Calculate Similarity Between Other Language Document Representations and Target Language Document Representations """
        similarities = torch.bmm(other_lang_representations_normed,target_lang_representations_normed).squeeze(-1) # B x M
        
        return similarities

    def forward(self,text_indices,sentence_lens,languages,document_level_text_indices,document_level_sentence_lens,token2id_dict,id2embedding_dict,phase,device):
        """ Forward Pass Through Text Encoder and Text Decoder """
        
        encoder_text_indices = text_indices['other']
        encoder_languages = languages['other']
        encoder_sentence_lens = sentence_lens['other']
        
        """ Use Token Representations for Cross-Attention with Decoder Representations """
        #print(encoder_text_indices[:5,:])
        token_representations = self.encoder.forward(encoder_text_indices,encoder_sentence_lens,encoder_languages,token2id_dict,id2embedding_dict,phase,device) #outputs is B x S x Words
        
        """ Obtain Document Representations for Each Language """
        all_document_representations = dict()
        for dest_lang in document_level_text_indices.keys():
            encoder_text_indices = document_level_text_indices[dest_lang]
            #print(encoder_text_indices.shape)
            encoder_sentence_lens = document_level_sentence_lens[dest_lang]
            #print(encoder_sentence_lens)
            document_representations = self.encoder.get_embeddings(encoder_text_indices,encoder_sentence_lens,token2id_dict,id2embedding_dict[dest_lang],phase,device)
            all_document_representations[dest_lang] = document_representations
                
        """ Language Modelling Using Decoder to Predict Words in the Target Document Based on Source Document(s) """
        #target_lang = 'de' # 'pt' depending on the dataset (or we can randomize it make to sure randomize in pad_collate as well)
        if 'ptbxl' in self.dataset_name:
            target_lang = 'de'
        elif 'brazil' in self.dataset_name:
            target_lang = 'pt'

        """ Calculate Cosine Similarity Between Evidence and Target Document Representations """
        similarities = self.obtain_cosine_similarity_matrix(all_document_representations,target_lang)
        
        decoder_text_indices = text_indices[target_lang]
        #decoder_languages = languages[target_lang]
        decoder_sentence_lens = sentence_lens[target_lang]
        sampling = None #filler (not used during pre-training or fine-tuning)
        outputs = self.decoder.forward(decoder_text_indices,decoder_sentence_lens,token2id_dict[target_lang],id2embedding_dict[target_lang],target_lang,phase,sampling,device,annot_vectors=token_representations,similarities=similarities,document_level_sentence_lens=document_level_sentence_lens)
        #outputs = self.decoder.forward(decoder_text_indices,decoder_languages,decoder_sentence_lens,token_representations,similarities,id2embedding_dict,target_lang,device)
        
        return outputs, target_lang
        

#%%
class text_decoder_network(nn.Module):
    
    def __init__(self,embedding_dim,vocab,attn=False,MAX_LENGTH=45,noutputs=None,num_layers=1,decoder_type='lstm',dest_lang_list=['en'],target_token_selection='uniform',replacement_prob=0.15,goal='IC'):#,h0=None,c0=None):
        super(text_decoder_network,self).__init__()
        input_features = embedding_dim if attn is not None else embedding_dim #word embedding dims #keep as same dimension as hidden to allow for attention calculation
        hidden_features = embedding_dim
        
        if decoder_type == 'lstm':
            """ LSTM NETWORK """
            self.num_layers = num_layers
            network = nn.LSTM(input_features,hidden_features,num_layers,batch_first=True)
        elif decoder_type == 'transformer':
            """ TRANSFORMER NETWORK """
            decoder_layer = nn.TransformerDecoderLayer(d_model=input_features, nhead=5)
            network = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        """ Layer Prior to LSTM Input To Collapse Concatenation to Correct Dimension """
        linear_pre_lstm = nn.Linear(2*embedding_dim,embedding_dim) #project from concatenation to half the dimension

        if goal == 'Text_Supervised':
            linear = nn.Linear(hidden_features,noutputs)
            self.linear = linear
        elif goal == 'Language_Change_Detection':
            linear = nn.Linear(hidden_features,1) #simple BCELoss
            self.replacement_prob = replacement_prob
            self.target_token_selection = target_token_selection
            self.linear = linear
        elif goal == 'Language_Detection':
            linear = nn.Linear(hidden_features,len(dest_lang_list)+1) 
            self.linear = linear
            self.target_token_selection = target_token_selection
            self.replacement_prob = replacement_prob
        elif goal in ['IC','MLM']:
            """ Linear Output Layer for Each Language Because Each Language Has Different Number of Tokens """
            linear_dict = nn.ModuleDict()
            for dest_lang in dest_lang_list:
                linear = nn.Linear(hidden_features,len(vocab[dest_lang]))
                """ Added New - December 27th, 2020 """
                #linear = nn.Linear(hidden_features,len(vocab[dest_lang])-1) #might have to be len(XYZ) - 1
                """ End """
                linear_dict[dest_lang] = linear
            self.linear_dict = linear_dict
            self.replacement_prob = replacement_prob
        elif goal == 'ELECTRA':
            """ MLP For Binary Detection (for Discriminator) """
            linear_detection = nn.Linear(hidden_features,1) #simple BCELoss
            self.linear_detection = linear_detection
            """ MLP for Language Modelling (for Generator) """
            linear_dict = nn.ModuleDict()
            for dest_lang in dest_lang_list:
                linear = nn.Linear(hidden_features,len(vocab[dest_lang]))
                linear_dict[dest_lang] = linear
            self.linear_dict = linear_dict
            """ Generator Network (Smaller than Discriminator Network) """
            decoder_layer = nn.TransformerDecoderLayer(d_model=input_features, nhead=5)
            generator_network = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2)
            self.generator_network = generator_network
            self.replacement_prob = replacement_prob            

        """ Weights to Transform Initial Hidden States Before Being Fed Into Decoder """
        h0_linear = nn.Linear(hidden_features,hidden_features)
        c0_linear = nn.Linear(hidden_features,hidden_features)
#        if self.num_layers > 1:
#            h0_linear_layer2 = nn.Linear(embedding_dim,embedding_dim)
#            c0_linear_layer2 = nn.Linear(embedding_dim,embedding_dim)            
        
        relu = nn.ReLU()
        
        self.hidden_features = hidden_features
        self.network = network
        self.linear_pre_lstm = linear_pre_lstm
        self.attn = attn 
        self.MAX_LENGTH = MAX_LENGTH
        self.goal = goal
        self.dest_lang_list = dest_lang_list
        self.decoder_type = decoder_type
        self.noutputs = noutputs
        self.h0_linear = h0_linear
        self.c0_linear = c0_linear
        self.relu = relu
        
        if self.attn == 'concat_nonlinear':
            """ Weights to Calculate Attention Coefficients """
            self.linear_attn1a = nn.Linear(input_features,input_features//2)
            self.linear_attn1b = nn.Linear(input_features,input_features//2)
            self.linear_attn2 = nn.Linear(input_features//2,1) #map to attention coefficient 
            self.Tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

    def calculate_training_attn_coefs(self,embeddings,h,annot_vectors,image_regions,instance_attn_coefs=None,s=0):
        """ Calculate Attention Coefficients When in the Training Stage (i.e. Batched) """
        if self.attn is not None:
            """ In the Event of Multiple Layers in LSTM, Only Use First Layer Hidden States For Attention Calculation """
            if h.shape[0] > 1:
                h = h[0,:,:].unsqueeze(0) # 1xBxL
            """ Calculate Attention """
            if self.attn == 'direct_dot_product':                
                attn_coefs = torch.bmm(h.permute(1,0,2),annot_vectors) # 1xBxL X BxLx79 = Bx1x79
                #attn_coefs = torch.bmm(embedding,instance_annot_vectors) # 1x1xL X 1xLx79 = 1x1x79
                attn_coefs = F.softmax(attn_coefs,dim=-1) # Bx1x79
            elif self.attn == 'concat_nonlinear':
                modified_embedding = h.repeat(image_regions,1,1).permute(1,0,2) #Bx79xL 
                concat_vector = torch.cat((modified_embedding,annot_vectors.permute(0,2,1)),2) #Bx79x2L
                intermediate_coefs_a = self.Tanh(self.linear_attn1a(concat_vector)) #Bx79xL
                intermediate_coefs_b = self.sigmoid(self.linear_attn1b(concat_vector)) #Bx79xL
                intermediate_coefs = intermediate_coefs_a * intermediate_coefs_b #Bx79xL
                current_attn_coefs = self.linear_attn2(intermediate_coefs).permute(0,2,1) #Bx1x79
                attn_coefs = F.softmax(current_attn_coefs,dim=-1) # Bx1x79
            """ Modify LSTM Input """
            #instance_attn_coefs[:,s,:] = attn_coefs.squeeze(1) #128x79
            context_vector = torch.bmm(annot_vectors,attn_coefs.permute(0,2,1)).permute(0,2,1) # BxLx79 X Bx79x1 = Bx1xL 
            embeddings = torch.cat((embeddings,context_vector),2) # Bx1xL X Bx1xL = Bx1x2L 
            """ Added New October 21 - Project to Half Dimension, L """
            embeddings = self.linear_pre_lstm(embeddings) # Bx1xL
            
        return embeddings#, instance_attn_coefs
    
    def calculate_inference_attn_coefs(self,embedding,h,instance_annot_vectors,image_regions,instance_attn_coefs,s=0):
        """ Calculate Attention Coefficients When in the Inference Stage (i.e. Not Batched) """
        if self.attn is not None:
            """ In the Event of Multiple Layers in LSTM, Only Use First Layer Hidden States For Attention Calculation """
            if h.shape[0] > 1:
                h = h[0,:,:].unsqueeze(0) # 1x1xL
            """ Calculate Attention """
            if self.attn == 'direct_dot_product':
                """ Modification October 15th, 2020 - Previous Hidden Vector to Calculate Attention (Not Embedding) """
                attn_coefs = torch.bmm(h,instance_annot_vectors) # 1x1xL X 1xLx79 = 1x1x79
                #attn_coefs = torch.bmm(embedding,instance_annot_vectors) # 1x1xL X 1xLx79 = 1x1x79
                attn_coefs = F.softmax(attn_coefs,dim=-1)
            elif self.attn == 'concat_nonlinear':
                modified_embedding = embedding.repeat(1,image_regions,1) #1x79xL
                concat_vector = torch.cat((modified_embedding,instance_annot_vectors.permute(0,2,1)),2) #1x79x2L
                intermediate_coefs_a = self.Tanh(self.linear_attn1a(concat_vector)) #1x79xL
                intermediate_coefs_b = self.sigmoid(self.linear_attn1b(concat_vector)) #1x79xL
                intermediate_coefs = intermediate_coefs_a * intermediate_coefs_b
                current_attn_coefs = self.linear_attn2(intermediate_coefs).permute(0,2,1) #1x1x79
                attn_coefs = F.softmax(current_attn_coefs,dim=-1)                    
            """ Modify LSTM Input """
            instance_attn_coefs[0,s,:] = attn_coefs 
            context_vector = torch.bmm(instance_annot_vectors,attn_coefs.permute(0,2,1)).permute(0,2,1) # 1xLx79 X 1x79x1 = 1x1xL
            embedding = torch.cat((embedding,context_vector),2) #1x1x2L
            """ Added New October 21 - Project to Half Dimension """
            embedding = self.linear_pre_lstm(embedding) # Bx1xL
            
        return embedding, instance_attn_coefs

    def transform_initial_hidden_states(self,h0,c0):
        #h0 = NUM_LAYERS x B x L
        h = self.h0_linear(h0.permute(1,0,2)).permute(1,0,2).contiguous()
        c = self.c0_linear(c0.permute(1,0,2)).permute(1,0,2).contiguous()
        return h, c

    def generate_mask_to_avoid_future_entries(self, output_sequence_length, device):#, sentence_lens):
        """ Mask To Avoid Looking at Future Entries """
        mask = (torch.triu(torch.ones(output_sequence_length, output_sequence_length,device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #""" Mask to Avoid Looking At Padded Entries When Calculating Attention """
        #masks = mask.unsqueeze(0).repeat(len(sentence_lens),1,1) # BxSxS
        #for i in range(masks.shape[0]):
        #    """ Create Padding Mask - i.e. Avoid Entries That Are Padded When Calculating Attention """
        #    padding_mask = torch.zeros_like(mask)
        #    padding_mask[:,sentence_lens[i]:] = float('-inf')
        #    """ Add Padding Mask to Future Entry Mask """
        #    masks[i,:,:] = masks[i,:,:] + padding_mask
        
        return mask

    def obtain_outputs_with_teacher_forcing(self,text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device):
        """ Convert Text Indices to List """
        nsentences = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
        all_outputs = torch.zeros(nsentences,ntokens-1,len(id2embedding_dict),device=device) # B x S x Words
#        all_attn_coefs = torch.zeros(nsentences,ntokens-1,image_regions,device=device) # B x S x #annot vectors
        
        if self.decoder_type == 'lstm':
            
            if h0 == None and c0 == None:
                """ OPTION 1 - Randomly-initialize Hidden Vectors """
                batch_size = len(text_indices)
                h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
                c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            else:
                """ Option 2 - Use Vectors from Encoder """
                h0 = h0.unsqueeze(0).repeat(self.num_layers,1,1) #1xBxL
                c0 = c0.unsqueeze(0).repeat(self.num_layers,1,1)
            
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            """ Pass Through LSTM Time-Step """
            for sentence_length in range(1,ntokens): #ntokens is the same for all sentences in the same batch due to padding introduced with dataloader
                token_indices = text_indices[:,sentence_length-1].tolist() # BxL
                embeddings = torch.stack([id2embedding_dict[token_index] for token_index in token_indices]) # BxL
                embeddings = embeddings.unsqueeze(1) # Bx1xL
                """ Calculation Modified Embeddings and Attention """
                #all_attn_coefs = for penultmiate argument
                embeddings = self.calculate_training_attn_coefs(embeddings,h,annot_vectors,image_regions,s=sentence_length-1)
                """ Forward Pass and Obtain Probabilities Over Next Word """
                outputs, (h, c) = self.network(embeddings,(h,c))
                outputs = self.linear_dict[dest_lang](outputs) #Bx1xV (V = Vocab)
                """ Store Predictions Over Words """
                all_outputs[:,sentence_length-1,:] = outputs.squeeze(1)
        elif self.decoder_type == 'transformer':
            """ Define Encoder Output For Use in Multi-head Attention """
            encoder_output = annot_vectors.permute(2,0,1) # BxLx79 
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS
            """ Convert Padded Sequences to Embeddings """ 
            decoder_input = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict)) for text_index in text_indices],0) #BxSxL
            """ Mask for Decoder Input to Avoid Looking Forward """
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS
            #print(decoder_input.permute(1,0,2).shape,encoder_output.shape,mask.shape)
            outputs = self.network(decoder_input.permute(1,0,2),encoder_output,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            all_outputs = self.linear_dict[dest_lang](outputs.permute(1,0,2)) #BxSxV (V = Vocab)
            """ Do Not Consider Output When Input is /END """
            all_outputs = all_outputs[:,:-1,:] #drop last sequence output
            #""" Filler for Attention Coefs """
            #all_attn_coefs = torch.empty(ntokens,ntokens)

        """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
        h0 = h0.detach()
        c0 = c0.detach()
                            
        """ Rename For Compatibility """
        #outputs_padded = all_outputs
        #attn_coefs = all_attn_coefs
        #sorted_indices = torch.arange(len(outputs_padded),device=device)  
        return all_outputs#, attn_coefs, sorted_indices        

    def obtain_outputs_without_teacher_forcing(self,text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device):
        """ List to Keep Track of Predicted Words """
        all_outputs = torch.zeros(len(text_indices),self.MAX_LENGTH,len(id2token_dict)) # B x S x L
        all_attn_coefs = torch.zeros(len(text_indices),self.MAX_LENGTH+1,image_regions) # B x S x #annot vectors
        
        if self.decoder_type == 'lstm':
            
            if h0 == None and c0 == None:
                """ OPTION 1 - Randomly-initialize Hidden Vectors """
                batch_size = len(text_indices)
                h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
                c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            else:
                """ Option 2 - Use Vectors from Encoder """
                h0 = h0.unsqueeze(0).repeat(self.num_layers,1,1) #1xBxL
                c0 = c0.unsqueeze(0).repeat(self.num_layers,1,1)
            
            """ Transform Initial Hidden State of Decoder """
            h0, c0 = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            """ Iterate Over Batch """
            for b in range(len(text_indices)):
                instance_outputs = torch.zeros(1,self.MAX_LENGTH,len(id2token_dict))
                instance_attn_coefs = torch.zeros(1,self.MAX_LENGTH+1,image_regions)
                """ Current Instance Hidden States """
                h = h0[:,b,:].unsqueeze(1).contiguous() #.view(self.num_layers,1,-1) # NUM_LAYERS x B=1 x L # 1x1xL
                c = c0[:,b,:].unsqueeze(1).contiguous() #.view(self.num_layers,1,-1) # NUM_LAYERS x B=1 x L # 1x1xL
                """ Get Current Sentence """
                current_text_indices = text_indices[b].tolist()
                """ Get Current Word """
                current_text_index = current_text_indices[0] #should be index associated with '/START' for all sentences in batch
                current_token = id2token_dict[current_text_index]
                """ Current Embedding """
                embedding = id2embedding_dict[current_text_index]
                """ Change Dimension of Embedding """
                embedding = embedding.view(1,1,-1) #1x1xL
                """ Calculate Context Vector Based on Attention (Now Called Embedding Which is Already Concatenated) """
                instance_annot_vectors = annot_vectors[b,:,:].unsqueeze(0)
                embedding, instance_attn_coefs = self.calculate_inference_attn_coefs(embedding,h,instance_annot_vectors,image_regions,instance_attn_coefs,s=0)
                """ Pass Through LSTM Time-Step """
                sentence_length = 0
                end_token = end_dict[dest_lang].lower()
                while (current_token != end_token) and (sentence_length < self.MAX_LENGTH): 
                    """ Forward Pass and Obtain Probabilities Over Next Word """
                    outputs, (h, c) = self.network(embedding,(h,c))
                    outputs = self.linear_dict[dest_lang](outputs) #1x1xV (V = Vocab)
                    
                    """ **** PREPARE FOR NEXT TIME-STEP **** """
                    
                    """ Obtain Predicted Word """
                    current_text_index = torch.argmax(outputs).item()
                    current_token = id2token_dict[current_text_index]  
                    """ Obtain Embedding of Predicted Word """
                    embedding = id2embedding_dict[current_text_index]
                    embedding = embedding.view(1,1,-1) #1x1xL  
                    """ Calculate Context Vector Based on Attention """
                    embedding, instance_attn_coefs = self.calculate_inference_attn_coefs(embedding,h,instance_annot_vectors,image_regions,instance_attn_coefs,s=sentence_length+1)
                    """ Store Predictions Over Words """
                    instance_outputs[0,sentence_length,:] = outputs
                    sentence_length += 1
                """ Store Predicted Words for This Sentence """
                all_outputs[b,:,:] = instance_outputs
                all_attn_coefs[b,:,:] = instance_attn_coefs.cpu().detach().numpy()
        elif self.decoder_type == 'transformer':
            for b in range(len(text_indices)):
                instance_outputs = torch.zeros(1,self.MAX_LENGTH,len(id2token_dict))
                #instance_attn_coefs = torch.zeros(1,self.MAX_LENGTH+1,image_regions)
                """ Get Current Sentence """
                current_text_indices = text_indices[b].tolist()
                """ Get Current Word """
                current_text_index = current_text_indices[0] #should be index associated with '/START' for all sentences in batch
                current_token = id2token_dict[current_text_index]
                """ Current Embedding """
                embedding = id2embedding_dict[current_text_index]
                """ Change Dimension of Embedding """
                embedding = embedding.view(1,1,-1) #1x1xL
                """ Calculate Context Vector Based on Attention (Now Called Embedding Which is Already Concatenated) """
                instance_annot_vectors = annot_vectors[b,:,:].unsqueeze(0)
                """ Pass Through LSTM Time-Step """
                sentence_length = 0
                embedding_inputs = embedding
                end_token = end_dict[dest_lang].lower()
                while (current_token != end_token) and (sentence_length < self.MAX_LENGTH): 
                    #print(current_token + ' ', end_token)
                    #print(current_token == end_token)
                    """ Forward Pass and Obtain Probabilities Over Next Word """
                    decoder_input = embedding_inputs # 1 (growing) x1xL
                    encoder_output = instance_annot_vectors.permute(2,0,1) # 79x1xL
                    mask = self.generate_mask_to_avoid_future_entries(sentence_length+1,device)
                    #print(decoder_input.shape,encoder_output.shape,mask.shape)
                    outputs = self.network(decoder_input,encoder_output,mask)
                    outputs = self.linear_dict[dest_lang](outputs) #1 (growing) x1xV (V = Vocab)
                    """ Once Sequence Grows, You Only Want Last Output """
                    outputs = outputs[-1,:,:] # 1xV
                    
                    """ **** PREPARE FOR NEXT TIME-STEP **** """
                    
                    """ Obtain Predicted Word """
                    current_text_index = torch.argmax(outputs).item()
                    current_token = id2token_dict[current_text_index]  
                    """ Obtain Embedding of Predicted Word """
                    embedding = id2embedding_dict[current_text_index]
                    embedding = embedding.view(1,1,-1) #1x1xL  
                    """ Grow Your Embedding Inputs """
                    embedding_inputs = torch.cat((embedding_inputs,embedding),0) # S (growing) x 1 x L
                    """ Store Predictions Over Words """
                    instance_outputs[0,sentence_length,:] = outputs
                    sentence_length += 1
                """ Store Predicted Words for This Sentence """
                all_outputs[b,:,:] = instance_outputs 
                #all_attn_coefs[b,:,:] = instance_attn_coefs
            
        """ Rename For Compatibility """
        #outputs_padded = all_outputs
        #attn_coefs = all_attn_coefs
        sorted_indices = 5#torch.arange(len(outputs_padded),device=device)  
        return all_outputs, all_attn_coefs, sorted_indices

    def categorical_token_sampling(self,replacement_langs,replacement_rows,replacement_cols,embeddings,id2embedding_dict):
        """ Sample Target Tokens from Target Language Using Categorical Distribution Based on Similarity Between Source Token and Target Tokens """
        replacement_ids = []
        for target_lang,replacement_row,replacement_col in zip(replacement_langs,replacement_rows,replacement_cols):
            """ Embedding to Replace """
            embedding_to_replace = embeddings[replacement_row,replacement_col,:].unsqueeze(0) # 1 x L
            """ Target Embeddings """
            target_embeddings = torch.stack(list(id2embedding_dict[target_lang].values()),0) # N x L
            """ Normalize Embeddings """
            embedding_to_replace_norm = torch.norm(embedding_to_replace) #scalar
            target_embeddings_norm = torch.norm(target_embeddings,dim=1).unsqueeze(1) # N x 1
            embedding_to_replace_normalized = torch.div(embedding_to_replace,embedding_to_replace_norm) # 1 x L
            target_embeddings_normalized = torch.div(target_embeddings,target_embeddings_norm) # N x L
            """ Calculate Similarities """
            similarities = torch.mm(embedding_to_replace_normalized,target_embeddings_normalized.transpose(0,1)).squeeze(0) # 1xL X LxN = 1xN = N
            """ Initialize Categorical Distribution """
            m = torch.distributions.categorical.Categorical(logits = similarities)
            """ Sample TokenID """
            current_id = m.sample([1]).item() - 1 #bc of -1 to len(X) - 1 range, due to masking token given -1 ID
            """ Append Current ID """
            replacement_ids.append(current_id)
        return replacement_ids

    def replace_embeddings_language_detection(self,embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device):
        """ Identify Replacement Languages """
        probability_of_replacement = self.replacement_prob #0.15
        """ Initialize Replacement Labels With All Zeros """
        replacement_label = np.zeros((batch_size,ntokens))
        padding_label = np.zeros((batch_size,ntokens))
        """ Sample From Uniform Distribution WITHIN Sentence Length and Populate Replacement Labels """
        for i,sentence_len in enumerate(sentence_lens):
            replacement_cols = np.where(np.random.uniform(0,1,(sentence_len)) < probability_of_replacement)#[0]
            replacement_label[i,replacement_cols] = 1
            padding_label[i,sentence_len:] = 1
        """ Create Matrix of Other Languages """
        nlanguages = len(list(token2id_dict.keys())) 
        language_to_label_dict = dict(zip(list(token2id_dict.keys()),np.arange(1,nlanguages+1)))
        remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
        other_lang = [[random.sample(remaining_languages,1)[0] for _ in range(ntokens)] for _ in range(batch_size)] # B x S-1
        #""" Convert Matrix of Other Languages to Matrix of Ground-Truth Labels """
        #language_replacement_labels = np.array([[language_to_label_dict[lang] for lang in row_lang] for row_lang in other_lang])#.reshape(batch_size,ntokens)
        #all_replacements = torch.tensor(language_replacement_labels,dtype=torch.long,device=device) # B x S-1 #we need zero entryies to act likke padding
        """ Obtain Padding Rows and Cols """
        padding_rows,padding_cols = np.where(padding_label == 1)
        """ Obtain Replacement Rows and Cols """
        replacement_rows,replacement_cols = np.where(replacement_label == 1) # K
        """ Obtain Languages at These Replacement Rows and Cols """
        replacement_langs = [other_lang[row_index][col_index] for row_index,col_index in zip(replacement_rows,replacement_cols)] # K
        """ Convert Languages to Labels at These Replacement Rows and Cols """
        replacement_labels = [language_to_label_dict[lang] for lang in replacement_langs]
        """ Initialize Ground-Truth Matrix With Source Language Label """
        all_replacements = np.ones_like(replacement_label)*language_to_label_dict[dest_lang]
        """ Assign New Labels """
        all_replacements[replacement_rows,replacement_cols] = replacement_labels
        """ Assign Padding Label """
        all_replacements[padding_rows,padding_cols] = 0
        """ Convert to Tensor """
        all_replacements = torch.tensor(all_replacements,dtype=torch.long,device=device)
        #print(all_replacements)
        """ Obtain Target Token IDs """
        if self.target_token_selection == 'categorical':
            replacement_ids = self.categorical_token_sampling(replacement_langs,replacement_rows,replacement_cols,embeddings,id2embedding_dict)
        elif self.target_token_selection == 'uniform':
            #replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs] # K 
            replacement_ids = [np.random.randint(0,len(id2embedding_dict[lang])-1,1).item() for lang in replacement_langs]

        """ Obtain Target Token Embeddings """
        replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) # K #nreplacements x L
        """ Replace Source Token Embeddings with Target Token Embeddings """
        embeddings[replacement_rows,replacement_cols,:] = replacement_embeddings #look out for this assignment
        return embeddings, all_replacements

    def language_detection_forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device):
        """ Forward Pass Where Tokens Are Replaced With Those from Other Languages and Task is to Predict Other Language 
        Args:
            token2id_dict (dict of dicts) lang -> token -> id
            idembedding_dict (dict of dicts) lang -> id -> embedding
        """
        batch_size = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
#        all_representations = torch.zeros(batch_size,ntokens-1,self.hidden_features,device=device) # B x S x Words
        all_outputs = torch.zeros(batch_size,ntokens-1,len(self.dest_lang_list),device=device) # B x S x C
        all_replacements = torch.zeros(batch_size,ntokens-1,device=device) # B x S
        
        if self.decoder_type == 'lstm':
            """ Initialize Hidden States """
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L

            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings of Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_language_detection(embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device)
            
            representations, (h, c) = self.network(embeddings,(h,c)) # h = B x L
            all_outputs = self.linear(representations) #BxSx1  
            """ Store Predictions Over Words """
#            all_representations = representations

#            """ Pass Through LSTM Time-Step """
#            for sentence_length in range(1,ntokens): #notkens is the same for all sentences in the same batch due to padding introduced with dataloader
#                token_indices = text_indices[:,sentence_length-1].tolist() # B
#                embeddings = torch.stack([id2embedding_dict[dest_lang][token_index] for token_index in token_indices]) # BxL
#                
#                """ Identify Replacement Languages """
#                probability_of_replacement = 0.15
#                replacement_label = np.where(np.random.uniform(0,1,batch_size) < probability_of_replacement,1,0)
#                nlanguages = len(list(token2id_dict.keys())) 
#                language_to_label_dict = dict(zip(list(token2id_dict.keys()),np.arange(nlanguages)))
#                remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
#                other_lang = [random.sample(remaining_languages,1)[0] for _ in range(batch_size)]
#                """ Obtain Language Label as Ground-Truth For Loss Calculation Later """
#                language_replacement_labels = [language_to_label_dict[lang] for lang in other_lang]
#                all_replacements[:,sentence_length-1] = torch.tensor(language_replacement_labels,dtype=torch.long)
#                """ If Replacement Needs to be Made """
#                if sum(replacement_label) > 0:
#                    """ Identify Tokens to Replace """
#                    replacement_indices = np.where(replacement_label == 1)[0]
#                    replacement_langs = [other_lang[index] for index in replacement_indices]
#                    """ Obtain Random Token From Other Language (More Sophisticated Strategy Can Also Be Used) """
#                    replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs]
#                    replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) #nreplacements x L
#                    embeddings[replacement_indices,:] = replacement_embeddings #look out for this assignment
#                
#                embeddings = embeddings.unsqueeze(1) # Bx1xL
#                """ Forward Pass and Obtain Probabilities Over Next Word """
#                outputs, (h, c) = self.network(embeddings,(h,c)) # h = BxLnew
#                outputs = self.linear(outputs) #Bx1x1 
#                """ Store Predictions Over Words """
#                all_representations[:,sentence_length-1,:] = h[-1,:] #final layer representation
#                all_outputs[:,sentence_length-1,:] = outputs.squeeze(1)
            
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach()
        elif self.decoder_type == 'transformer':            
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_language_detection(embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device)
            
            embeddings = embeddings.permute(1,0,2) # S x B x L
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS
            representations = self.network(embeddings,embeddings,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            all_outputs = self.linear(representations.permute(1,0,2)) #BxSxC
            """ Do Not Consider Output When Input is /END """
            #all_outputs = all_outputs[:,:,:] #drop last sequence output
#            all_representations = representations.permute(1,0,2)

#            for b in range(len(text_indices)):
#                """ Obtain All Tokens In Sentence Except for Last One """
#                sentence_indices = text_indices[b,:-1].tolist() #S 
#                """ Obtain Embeddings """
#                embeddings = torch.stack([id2embedding_dict[dest_lang][token_index] for token_index in sentence_indices]) # SxL
#                
#                """ Identify Replacement Languages """
#                ntokens = embeddings.shape[0]
#                probability_of_replacement = 0.15
#                replacement_label = np.where(np.random.uniform(0,1,ntokens) < probability_of_replacement,1,0)
#                nlanguages = len(list(token2id_dict.keys())) 
#                language_to_label_dict = dict(zip(list(token2id_dict.keys()),np.arange(nlanguages)))
#                remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
#                other_lang = [random.sample(remaining_languages,1)[0] for _ in range(ntokens)]
#                """ Obtain Language Label as Ground-Truth For Loss Calculation Later """
#                language_replacement_labels = [language_to_label_dict[lang] for lang in other_lang]
#                all_replacements[b,:] = torch.tensor(language_replacement_labels,dtype=torch.long)
#                """ If Replacement Needs to be Made """
#                if sum(replacement_label) > 0:
#                    """ Identify Tokens to Replace """
#                    replacement_indices = np.where(replacement_label == 1)[0]
#                    replacement_langs = [other_lang[index] for index in replacement_indices]
#                    """ Obtain Random Token From Other Language (More Sophisticated Strategy Can Also Be Used) """
#                    replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs]
#                    replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) #nreplacements x L
#                    embeddings[replacement_indices,:] = replacement_embeddings #look out for this assignment
#                
#                embeddings = embeddings.unsqueeze(1) # Sx1xL
#                sentence_len = sentence_lens[b] #scalar 
#                """ Create Mask for Padded Components (Do Not Include Padded Components in Self-Attention Calculations) """
#                ntokens = len(text_indices[0])-1
#                mask = torch.ones(ntokens,ntokens,device=device)*(float('-inf'))
#                mask[:,:sentence_len] = torch.tensor(0)
#                """ Obtain Representations """
#                representations = self.network(embeddings,embeddings,mask) #expects Sx1xL
#                """ Obtain Posterior Class Distribution """
#                outputs = self.linear(representations) #Sx1x1
#                
#                all_representations[b,:,:] = representations.squeeze(1)
#                all_outputs[b,:,:] = outputs.squeeze(1)

        """ Rename For Compatibility """
#        representations_padded = all_representations
        outputs_padded = all_outputs
        replacement_labels_padded = all_replacements

#        sentence_lengths = torch.tensor(sentence_lens,device=device)#.index_select(0,sorted_indices)
#        if self.decoder_type == 'lstm':
#            """ Take All Entries into Consideration """
#            representations_padded = [sequence_representations[:sentence_length] for sequence_representations,sentence_length in zip(representations_padded,sentence_lengths)]
#            outputs_padded = [sequence_predictions[:sentence_length] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]
#            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
#        elif self.decoder_type == 'transformer':
#            """ Obtain All Sequence Outputs Over Classes Per Sentence """
#            outputs_padded = [sequence_predictions[:sentence_length,:] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]        
#            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
        
        """ Detach Items From Graph """
#        all_representations = all_representations.detach()
        all_outputs = all_outputs.detach()
        all_replacements = all_replacements.detach()

        return outputs_padded, replacement_labels_padded

    def replace_embeddings_language_change_detection(self,embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device):
        """ Identify Replacement Languages """
        #ntokens = embeddings.shape[1]
        probability_of_replacement = self.replacement_prob #0.15

        """ Initialize Replacement Labels With All Zeros """
        replacement_label = np.zeros((batch_size,ntokens))
        #padding_label = np.zeros((batch_size,ntokens))
        """ Sample From Uniform Distribution WITHIN Sentence Length and Populate Replacement Labels """
        for i,sentence_len in enumerate(sentence_lens):
            replacement_cols = np.where(np.random.uniform(0,1,(sentence_len)) < probability_of_replacement)#[0]
            replacement_label[i,replacement_cols] = 1
            #padding_label[i,sentence_len:] = 1

        replacement_label = np.where(np.random.uniform(0,1,(batch_size,ntokens)) < probability_of_replacement,1,0)
        remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
        other_lang = [[random.sample(remaining_languages,1)[0] for _ in range(ntokens)] for _ in range(batch_size)] # B x S-1
        """ Obtain Language Label as Ground-Truth For Loss Calculation Later """
        all_replacements = torch.tensor(replacement_label,dtype=torch.float,device=device).unsqueeze(2) # B x S-1 x 1
        """ Identify Tokens to Replace """
        replacement_rows,replacement_cols = np.where(replacement_label == 1) # K
        replacement_langs = [other_lang[row_index][col_index] for row_index,col_index in zip(replacement_rows,replacement_cols)] # K

        """ Obtain Target Token IDs """
        if self.target_token_selection == 'categorical':
            replacement_ids = self.categorical_token_sampling(replacement_langs,replacement_rows,replacement_cols,embeddings,id2embedding_dict)
        elif self.target_token_selection == 'uniform':
            #replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs] # K 
            replacement_ids = [np.random.randint(0,len(id2embedding_dict[lang])-1,1).item() for lang in replacement_langs]
            #replacement_ids = np.random.randint(0,len(id2embedding_dict[dest_lang])-1,len(replacement_langs)) #with replacement 
        
        replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) # K #nreplacements x L
        embeddings[replacement_rows,replacement_cols,:] = replacement_embeddings #look out for this assignment
        return embeddings, all_replacements

    def language_change_detection_forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device):
        """ Forward Pass Where Tokens Are Replaced With Those from Other Languages and Binary Prediction is Made 
        Args:
            token2id_dict (dict of dicts) lang -> token -> id
            idembedding_dict (dict of dicts) lang -> id -> embedding
        """
        batch_size = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
#        all_representations = torch.zeros(batch_size,ntokens-1,self.hidden_features,device=device) # B x S x Words
        all_outputs = torch.zeros(batch_size,ntokens-1,1,device=device) # B x S x C
        all_replacements = torch.zeros(batch_size,ntokens-1,1,device=device) # B x S
        
        if self.decoder_type == 'lstm':
            """ Initialize Hidden States """
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]                      
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_language_change_detection(embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device)            
            
            representations, (h, c) = self.network(embeddings,(h,c)) # h = B x L
            all_outputs = self.linear(representations) #BxSx1  
            """ Store Predictions Over Words """
#            all_representations = representations # B x S x L #final layer representation
            #all_outputs = outputs
            
#            """ Pass Through LSTM Time-Step """
#            for sentence_length in range(1,ntokens): #notkens is the same for all sentences in the same batch due to padding introduced with dataloader
#                token_indices = text_indices[:,sentence_length-1].tolist() # B
#                embeddings = torch.stack([id2embedding_dict[dest_lang][token_index] for token_index in token_indices]) # BxL
#                
#                """ Identify Replacement Languages """
#                probability_of_replacement = 0.15
#                replacement_label = np.where(np.random.uniform(0,1,batch_size) < probability_of_replacement,1,0)
#                all_replacements[:,sentence_length-1,:] = torch.tensor(replacement_label,dtype=torch.float).unsqueeze(1)
#                remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
#                other_lang = [random.sample(remaining_languages,1)[0] for _ in range(batch_size)]
#                """ If Replacement Needs to be Made """
#                if sum(replacement_label) > 0:
#                    """ Identify Tokens to Replace """
#                    replacement_indices = np.where(replacement_label == 1)[0]
#                    replacement_langs = [other_lang[index] for index in replacement_indices]
#                    """ Obtain Random Token From Other Language (More Sophisticated Strategy Can Also Be Used) """
#                    replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs]
#                    replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) #nreplacements x L
#                    embeddings[replacement_indices,:] = replacement_embeddings #look out for this assignment
#                
#                embeddings = embeddings.unsqueeze(1) # Bx1xL
#                """ Forward Pass and Obtain Probabilities Over Next Word """
#                outputs, (h, c) = self.network(embeddings,(h,c)) # h = BxLnew
#                outputs = self.linear(outputs) #Bx1x1 
#                """ Store Predictions Over Words """
#                all_representations[:,sentence_length-1,:] = h[-1,:] #final layer representation
#                all_outputs[:,sentence_length-1,:] = outputs.squeeze(1)
            
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach()
        elif self.decoder_type == 'transformer':    
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_language_change_detection(embeddings,sentence_lens,ntokens,batch_size,token2id_dict,id2embedding_dict,dest_lang,device)            
                        
            embeddings = embeddings.permute(1,0,2) # S x B x L
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS            
            representations = self.network(embeddings,embeddings,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            all_outputs = self.linear(representations.permute(1,0,2)) #BxSx1 
            """ Do Not Consider Output When Input is /END """
            #all_outputs = all_outputs[:,:,:] #drop last sequence output
#            all_representations = representations.permute(1,0,2) #BxSxL 
            
#            for b in range(len(text_indices)):
#                """ Obtain All Tokens In Sentence Except for Last One """
#                sentence_indices = text_indices[b,:-1].tolist() #S 
#                """ Obtain Embeddings """
#                embeddings = torch.stack([id2embedding_dict[dest_lang][token_index] for token_index in sentence_indices]) # SxL
#                
#                """ Identify Replacement Languages """
#                ntokens = embeddings.shape[0]
#                probability_of_replacement = 0.15
#                replacement_label = np.where(np.random.uniform(0,1,ntokens) < probability_of_replacement,1,0)
#                all_replacements[b,:,:] = torch.tensor(replacement_label,dtype=torch.float).unsqueeze(1)
#                remaining_languages = set(list(token2id_dict.keys())) - set([dest_lang])
#                other_lang = [random.sample(remaining_languages,1)[0] for _ in range(ntokens)]
#                """ If Replacement Needs to be Made """
#                if sum(replacement_label) > 0:
#                    """ Identify Tokens to Replace """
#                    replacement_indices = np.where(replacement_label == 1)[0]
#                    replacement_langs = [other_lang[index] for index in replacement_indices]
#                    """ Obtain Random Token From Other Language (More Sophisticated Strategy Can Also Be Used) """
#                    replacement_ids = [random.sample(list(range(len(id2embedding_dict[lang]))),1)[0] for lang in replacement_langs]
#                    replacement_embeddings = torch.stack([id2embedding_dict[lang][id_entry] for lang,id_entry in zip(replacement_langs,replacement_ids)]) #nreplacements x L
#                    embeddings[replacement_indices,:] = replacement_embeddings #look out for this assignment
#                
#                embeddings = embeddings.unsqueeze(1) # Sx1xL
#                sentence_len = sentence_lens[b] #scalar 
#                """ Create Mask for Padded Components (Do Not Include Padded Components in Self-Attention Calculations) """
#                ntokens = len(text_indices[0])-1
#                mask = torch.ones(ntokens,ntokens,device=device)*(float('-inf'))
#                mask[:,:sentence_len] = torch.tensor(0)
#                """ Obtain Representations """
#                representations = self.network(embeddings,embeddings,mask) #expects Sx1xL
#                """ Obtain Posterior Class Distribution """
#                outputs = self.linear(representations) #Sx1x1
#                
#                all_representations[b,:,:] = representations.squeeze(1)
#                all_outputs[b,:,:] = outputs.squeeze(1)

        """ Rename For Compatibility """
#        representations_padded = all_representations
        outputs_padded = all_outputs
        replacement_labels_padded = all_replacements

        sentence_lengths = torch.tensor(sentence_lens,device=device)#.index_select(0,sorted_indices)
        if self.decoder_type == 'lstm':
            """ Take All Entries into Consideration """
#            representations_padded = [sequence_representations[:sentence_length] for sequence_representations,sentence_length in zip(representations_padded,sentence_lengths)]
            outputs_padded = [sequence_predictions[:sentence_length] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]
            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
        elif self.decoder_type == 'transformer':
            """ Obtain All Sequence Outputs Over Classes Per Sentence """
            outputs_padded = [sequence_predictions[:sentence_length,:] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]        
            replacement_labels_padded = [sequence_labels[:sentence_length,:] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
        
        """ Detach Items From Graph """
#        all_representations = all_representations.detach()
        all_outputs = all_outputs.detach()
        all_replacements = all_replacements.detach()

        return outputs_padded, replacement_labels_padded

    def replace_embeddings_with_mask_token(self,embeddings,sentence_lens,ntokens,batch_size,id2embedding_dict,dest_lang,device):
        """ Identify Replacement Languages """
        probability_of_replacement = self.replacement_prob #0.15
        """ Initialize Replacement Labels With All Zeros """
        replacement_label = np.zeros((batch_size,ntokens))
        #replacement_label = np.zeros_like(embeddings[:,:,0])
        #padding_label = np.zeros((batch_size,ntokens))
        
        probability_of_masking = 0.80
        probability_of_random_token = 0.10
        #probability_of_no_replacement = 0.10
        
        #probability_of_mask = probability_of_mask * probability_of_replacement # 0.80 * 0.15
        #probability_of_random_token = probability_of_random_token * probability_of_replacement # 0.10 * 0.15
        
        """ Sample From Uniform Distribution WITHIN Sentence Length and Populate Replacement Labels """
        for i,sentence_len in enumerate(sentence_lens):
            random_for_replacement = np.random.uniform(0,1,(sentence_len)) # compare to probability of replacement
            random_for_masking = np.random.uniform(0,1,(sentence_len)) # compare to probability of mask
            """ Determine Which Tokens Need to Be Earmarked for Potential Masking/Replacement """
            replacement_bool = random_for_replacement < probability_of_replacement
            """ Determine Which Tokens Need to Be Masked """
            masking_bool = random_for_masking < probability_of_masking
            masking_cols = np.where(replacement_bool & masking_bool)
            replacement_label[i,masking_cols] = 1
            """ Determine Which Tokens Needs to Be Replaced Randomly """
            random_token_bool1 = random_for_masking <= (probability_of_masking + probability_of_random_token)
            random_token_bool2 = random_for_masking > probability_of_masking
            random_token_cols = np.where(replacement_bool & random_token_bool1 & random_token_bool2)
            replacement_label[i,random_token_cols] = 2
        
        """ Obtain Masking Rows and Cols """
        masking_rows,masking_cols = np.where(replacement_label == 1) # K
        """ Obtain Target Token Embeddings """ #-1 represents the mask
        masking_embeddings = torch.stack([id2embedding_dict[dest_lang][-1] for _ in masking_rows]) # K #nreplacements x L
        """ Replace Source Token Embeddings with Target Token Embeddings """
        embeddings[masking_rows,masking_cols,:] = masking_embeddings #look out for this assignment

        """ Obtain Random Token Rows and Cols """ #-1 represents the mask
        random_token_rows,random_token_cols = np.where(replacement_label == 2) # K
        """ Only Replace if Something Needs to Be Replaced """
        if len(random_token_rows) > 0:
            """ Because Mask = -1 ID, We Sample From 0 to Len(A) - 1 """
            random_token_ids = np.random.randint(0,len(id2embedding_dict[dest_lang])-1,len(random_token_rows)) #with replacement 
            """ Obtain Random Token Embeddings """
            random_token_embeddings = torch.stack([id2embedding_dict[dest_lang][id_entry] for id_entry in random_token_ids]) # K #nreplacements x L
            """ Replace Source Token Embeddings with Target Token Embeddings """
            embeddings[random_token_rows,random_token_cols,:] = random_token_embeddings #look out for this assignmen

        """ Convert to Tensor """
        all_replacements = torch.tensor(replacement_label,dtype=torch.float,device=device)

        return embeddings, all_replacements

    def MLM_forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,device):
        batch_size = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
        all_outputs = torch.zeros(batch_size,ntokens-1,len(self.dest_lang_list),device=device) # B x S x C
        all_replacements = torch.zeros(batch_size,ntokens-1,device=device) # B x S
        
        if self.decoder_type == 'lstm':
            """ Initialize Hidden States """
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L

            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings of Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_with_mask_token(embeddings,sentence_lens,ntokens,batch_size,id2embedding_dict,dest_lang,device)
            
            representations, (h, c) = self.network(embeddings,(h,c)) # h = B x L
            all_outputs = self.linear_dict[dest_lang](representations) #BxSxC
#            """ Store Predictions Over Words """
#            all_representations = representations
            
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach()
        elif self.decoder_type == 'transformer':            
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            embeddings, all_replacements = self.replace_embeddings_with_mask_token(embeddings,sentence_lens,ntokens,batch_size,id2embedding_dict,dest_lang,device)
            
            embeddings = embeddings.permute(1,0,2) # S x B x L
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS
            representations = self.network(embeddings,embeddings,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            all_outputs = self.linear_dict[dest_lang](representations.permute(1,0,2)) #BxSxC
            """ Do Not Consider Output When Input is /END """
            #all_outputs = all_outputs[:,:,:] #drop last sequence output

        """ Rename For Compatibility """
        #outputs_padded = all_outputs
        #replacement_labels_padded = all_replacements

#        sentence_lengths = torch.tensor(sentence_lens,device=device)#.index_select(0,sorted_indices)
#        if self.decoder_type == 'lstm':
#            """ Take All Entries into Consideration """
#            representations_padded = [sequence_representations[:sentence_length] for sequence_representations,sentence_length in zip(representations_padded,sentence_lengths)]
#            outputs_padded = [sequence_predictions[:sentence_length] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]
#            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
#        elif self.decoder_type == 'transformer':
#            """ Obtain All Sequence Outputs Over Classes Per Sentence """
#            outputs_padded = [sequence_predictions[:sentence_length,:] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]        
#            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
        
        """ Detach Items From Graph """
#        all_representations = all_representations.detach()
#        all_outputs = all_outputs.detach()
#        all_replacements = all_replacements.detach()
        return all_outputs, all_replacements#_labels_padded
        
    def obtain_outputs_with_ELECTRA_generator(self,text_indices,sentence_lens,id2embedding_dict,dest_lang,h0,c0,device):
        """ Convert Text Indices to List """
        nsentences = len(text_indices)
        text_indices = text_indices[:,:-1]
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
        all_outputs = torch.zeros(nsentences,ntokens-1,len(id2embedding_dict),device=device) # B x S x Words
        batch_size = len(text_indices)
        
        if self.decoder_type == 'lstm':
            
            if h0 == None and c0 == None:
                """ OPTION 1 - Randomly-initialize Hidden Vectors """
                h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
                c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            else:
                """ Option 2 - Use Vectors from Encoder """
                h0 = h0.unsqueeze(0).repeat(self.num_layers,1,1) #1xBxL
                c0 = c0.unsqueeze(0).repeat(self.num_layers,1,1)
            
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            """ Pass Through LSTM Time-Step """
            for sentence_length in range(1,ntokens): #ntokens is the same for all sentences in the same batch due to padding introduced with dataloader
                token_indices = text_indices[:,sentence_length-1].tolist() # BxL
                embeddings = torch.stack([id2embedding_dict[dest_lang][token_index] for token_index in token_indices]) # BxL
                embeddings = embeddings.unsqueeze(1) # Bx1xL
                """ Mask Tokens Randomly/As Per BERT """ 
                embeddings, all_replacements = self.replace_embeddings_with_mask_token(embeddings,sentence_lens,ntokens,batch_size,id2embedding_dict,dest_lang,device)
                """ Calculation Modified Embeddings and Attention """
                #all_attn_coefs = for penultmiate argument
                #embeddings = self.calculate_training_attn_coefs(embeddings,h,annot_vectors,image_regions,s=sentence_length-1)
                """ Forward Pass and Obtain Probabilities Over Next Word """
                outputs, (h, c) = self.generator_network(embeddings,(h,c))
                outputs = self.linear_dict[dest_lang](outputs) #Bx1xV (V = Vocab)
                """ Store Predictions Over Words """
                all_outputs[:,sentence_length-1,:] = outputs.squeeze(1)
        elif self.decoder_type == 'transformer':
#            """ Define Encoder Output For Use in Multi-head Attention """
#            encoder_output = annot_vectors.permute(2,0,1) # BxLx79 
            """ Obtain Text Indices in List Form """
            #text_indices = text_indices[:,:-1]
            text_indices = text_indices.tolist() # BxS
            """ Convert Padded Sequences to Embeddings """ 
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #BxSxL
            """ Mask Tokens Randomly/As Per BERT """
            embeddings, all_replacements = self.replace_embeddings_with_mask_token(embeddings,sentence_lens,ntokens,batch_size,id2embedding_dict,dest_lang,device)
            """ Mask for Decoder Input to Avoid Looking Forward """
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS
            #print(decoder_input.permute(1,0,2).shape,encoder_output.shape,mask.shape)
            embeddings = embeddings.permute(1,0,2)
            outputs = self.generator_network(embeddings,embeddings,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            all_outputs = self.linear_dict[dest_lang](outputs.permute(1,0,2)) #BxSxV (V = Vocab)
            #""" Do Not Consider Output When Input is /END """
            #all_outputs = all_outputs[:,:-1,:] #drop last sequence output
            #""" Filler for Attention Coefs """
            #all_attn_coefs = torch.empty(ntokens,ntokens)

        #""" Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
        #h0 = h0.detach()
        #c0 = c0.detach()
                            
        return all_outputs, all_replacements

    def ELECTRA_forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device,h0=None,c0=None):
        
        """ Perform Forward Pass Through Generator Using Masked Tokens """
        generator_outputs, all_replacements = self.obtain_outputs_with_ELECTRA_generator(text_indices,sentence_lens,id2embedding_dict,dest_lang,h0,c0,device)        
        """ all_replacements == 1 is the masked token locations """
        token_mask = torch.where(all_replacements == 1,torch.tensor(1,device=device),torch.tensor(0,device=device)).type(torch.bool) # K entries with a value 1
        """ Number of Masked Tokens """
        nmasks = torch.sum(token_mask)
        """ Broadcast Token Mask """
        token_mask_repeated = token_mask.unsqueeze(-1).repeat(1,1,len(id2embedding_dict[dest_lang])) # B x S x Nwords
        """ Obtain Outputs Corresponding to Masked Input Tokens """
        #print(generator_outputs.shape,token_mask_repeated.shape,nmasks,generator_outputs.masked_select(token_mask_repeated).shape)
        masked_outputs = generator_outputs.masked_select(token_mask_repeated).reshape(nmasks,-1) # K x Nwords
        """ Sample Tokens from Masked Outputs """
        m = torch.distributions.categorical.Categorical(logits = masked_outputs)
        sampled_tokenids = m.sample([1]).squeeze(0) - 1 # K   # -1 to account for mask ID = -1 (-1 to Len(A) - 1) #alternatively, change MASK ID to POS VALUE
        """ Check if TokenIds Match Ground-Truth """
        ground_truth_tokenids = text_indices[:,1:]
        masked_ground_truth_tokenids = ground_truth_tokenids.masked_select(token_mask) # K
        """ Modify Ground-Truth Labels for Discriminator """
        discriminator_token_mask = torch.zeros_like(token_mask,dtype=torch.float,device=device) # B x S
        discriminator_labels_flat = torch.where(sampled_tokenids != masked_ground_truth_tokenids,torch.tensor(1,dtype=torch.float,device=device),torch.tensor(0,dtype=torch.float,device=device))
        mask_rows, mask_cols = torch.where(token_mask) # K
        discriminator_token_mask[mask_rows,mask_cols] = discriminator_labels_flat #B x S
        """ Perform Forward Pass Through Discriminator Using Sampled Tokenids """
        
        batch_size = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
#        all_representations = torch.zeros(batch_size,ntokens-1,self.hidden_features,device=device) # B x S x Words
#        all_outputs = torch.zeros(batch_size,ntokens-1,1,device=device) # B x S x C
#        all_replacements = torch.zeros(batch_size,ntokens-1,1,device=device) # B x S
        
        if self.decoder_type == 'lstm':
            """ Initialize Hidden States """
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]                      
            """ Replace Embeddings """
            replacement_embeddings = torch.stack([id2embedding_dict[dest_lang][id_entry] for id_entry in sampled_tokenids.tolist()]) # K #nreplacements x L
            embeddings[mask_rows,mask_cols,:] = replacement_embeddings #look out for this assignment
            
            representations, (h, c) = self.network(embeddings,(h,c)) # h = B x L
            discriminator_outputs = self.linear_detection(representations) #BxSx1  
#            """ Store Predictions Over Words """
#            all_representations = representations # B x S x L #final layer representation        
            
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach()
        elif self.decoder_type == 'transformer':    
            text_indices = text_indices[:,:-1] # BxS-1
            """ Obtain Text Indices in List Form """
            text_indices = text_indices.tolist() # BxS-1
            """ Obtain Embeddings for Tokens """
            embeddings = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict[dest_lang])) for text_index in text_indices],0) #B x S-1 x L
            """ Determine Number of Tokens - Same for All Sentences Because of Padding """
            ntokens = embeddings.shape[1]
            """ Replace Embeddings """
            replacement_embeddings = torch.stack([id2embedding_dict[dest_lang][id_entry] for id_entry in sampled_tokenids.tolist()]) # K #nreplacements x L
            embeddings[mask_rows,mask_cols,:] = replacement_embeddings #look out for this assignment
                        
            embeddings = embeddings.permute(1,0,2) # S x B x L
            mask = self.generate_mask_to_avoid_future_entries(ntokens,device) # SxS            
            representations = self.network(embeddings,embeddings,mask) #input = SxBxL, input2 = 79xBxL, output = SxBxL
            """ Obtain Predictions Over Words """
            discriminator_outputs = self.linear_detection(representations.permute(1,0,2)) #BxSx1 
            """ Do Not Consider Output When Input is /END """
            #all_outputs = all_outputs[:,:,:] #drop last sequence output
#            all_representations = representations.permute(1,0,2) #BxSxL 

        """ Rename For Compatibility """
#        representations_padded = all_representations
#        outputs_padded = all_outputs
#        replacement_labels_padded = all_replacements

#        sentence_lengths = torch.tensor(sentence_lens,device=device)#.index_select(0,sorted_indices)
#        if self.decoder_type == 'lstm':
#            """ Take All Entries into Consideration """
##            representations_padded = [sequence_representations[:sentence_length] for sequence_representations,sentence_length in zip(representations_padded,sentence_lengths)]
#            outputs_padded = [sequence_predictions[:sentence_length] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]
#            replacement_labels = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(discriminator_token_mask,sentence_lengths)]
##            replacement_labels_padded = [sequence_labels[:sentence_length] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
#        elif self.decoder_type == 'transformer':
#            """ Obtain All Sequence Outputs Over Classes Per Sentence """
#            outputs_padded = [sequence_predictions[:sentence_length,:] for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)]        
##            replacement_labels_padded = [sequence_labels[:sentence_length,:] for sequence_labels,sentence_length in zip(replacement_labels_padded,sentence_lengths)]
        
        """ Detach Items From Graph """
#        all_representations = all_representations.detach()
#        all_outputs = all_outputs.detach()
#        all_replacements = all_replacements.detach()
        
        return generator_outputs, token_mask, discriminator_outputs, discriminator_token_mask

    def supervised_forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,phase,device):
        batch_size = len(text_indices)
        ntokens = len(text_indices[0]) #which is the same as ntokens when we pad the text indices in the dataloader section
#        all_representations = torch.zeros(batch_size,ntokens-1,self.hidden_features,device=device) # B x S x Words
        all_outputs = torch.zeros(batch_size,ntokens-1,self.noutputs,device=device) # B x S x C
        
        if self.decoder_type == 'lstm':
            """ Initialize Hidden States """
            h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
            c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
            """ Transform Initial Hidden State of Decoder """
            h, c = self.transform_initial_hidden_states(h0,c0) # NUM_LAYERS x B x L
            """ Pass Through LSTM Time-Step """
            for sentence_length in range(1,ntokens): #notkens is the same for all sentences in the same batch due to padding introduced with dataloader
                token_indices = text_indices[:,sentence_length-1].tolist() # BxS
                embeddings = torch.stack([id2embedding_dict[token_index] for token_index in token_indices]) # BxL
                embeddings = embeddings.unsqueeze(1) # Bx1xL
                """ Forward Pass and Obtain Probabilities Over Next Word """
                outputs, (h, c) = self.network(embeddings,(h,c)) # h = BxLnew
                outputs = self.linear(outputs) #Bx1xV (V = Vocab)
                """ Store Predictions Over Words """
#                all_representations[:,sentence_length-1,:] = h
                all_outputs[:,sentence_length-1,:] = outputs.squeeze(1)
            
            """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
            h0 = h0.detach()
            c0 = c0.detach()
        elif self.decoder_type == 'transformer':            
            for b in range(len(text_indices)):
                """ Obtain All Tokens In Sentence Except for Last One """
                sentence_indices = text_indices[b,:-1].tolist() #X
                """ Obtain Embeddings """
                embeddings = torch.stack([id2embedding_dict[token_index] for token_index in sentence_indices]) # SxL
                embeddings = embeddings.unsqueeze(1) # Sx1xL
                sentence_len = sentence_lens[b] #scalar 
                """ Create Mask for Padded Components (Do Not Include Padded Components in Self-Attention Calculations) """
                ntokens = len(text_indices[0])-1
                mask = torch.ones(ntokens,ntokens,device=device)*(float('-inf'))
                mask[:,:sentence_len] = torch.tensor(0)
                """ Obtain Representations """
                representations = self.network(embeddings,embeddings,mask) #expects Sx1xL
                """ Obtain Posterior Class Distribution """
                outputs = self.linear(representations) #Sx1xC 
                
#                all_representations[b,:,:] = representations.squeeze(1)
                all_outputs[b,:,:] = outputs.squeeze(1)

        """ Rename For Compatibility """
#        representations_padded = all_representations
        outputs_padded = all_outputs

        sentence_lengths = torch.tensor(sentence_lens,device=device)#.index_select(0,sorted_indices)
        if self.decoder_type == 'lstm':
            """ Final Entry in Sequence Output Only is Considered (Can be Changed to Take All Entries into Consideration) """
#            representations_padded = torch.cat([sequence_representations.index_select(0,sentence_length-1) for sequence_representations,sentence_length in zip(representations_padded,sentence_lengths)])
            outputs_padded = torch.cat([sequence_predictions.index_select(0,sentence_length-1) for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)])
        elif self.decoder_type == 'transformer':
            """ Take The Average of the Relevant Sequence Outputs To Obtain Single Distribution Over Classes Per Sentence """
            outputs_padded = torch.stack([torch.mean(sequence_predictions[:sentence_length,:],0) for sequence_predictions,sentence_length in zip(outputs_padded,sentence_lengths)])            
        
        """ Detach Items From Graph """
#        all_representations = all_representations.detach()
        all_outputs = all_outputs.detach()

        return outputs_padded

    def forward(self,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device,h0=None,c0=None,annot_vectors=None): #remember, this is at the batch level

        id2token_dict = {value:key for key,value in token2id_dict.items()}  
        image_regions = annot_vectors.shape[-1]

        if 'train' in phase:
            if sampling == False:
                """ Teacher Forcing """ #attn_coefs, sorted_indices
                outputs_padded = self.obtain_outputs_with_teacher_forcing(text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device)
            else:
                """ No Teacher Forcing (Obtain Argmax of Output Words Instead) """
                outputs_padded, attn_coefs, sorted_indices = self.obtain_outputs_without_teacher_forcing(text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device)
        else:
            """ No Teacher Forcing During Evaluation """
            outputs_padded, attn_coefs, sorted_indices = self.obtain_outputs_without_teacher_forcing(text_indices,sentence_lens,id2embedding_dict,id2token_dict,annot_vectors,dest_lang,image_regions,h0,c0,device)
        
        """ Detach Hidden States to Avoid Memory Leakage (No Longer Needed Anyway) """
        h0 = h0.detach()
        c0 = c0.detach()

        return outputs_padded #, sorted_indices, attn_coefs #

class combined_image_captioning_network(nn.Module):
    
    def __init__(self,encoder,decoder):
        super(combined_image_captioning_network,self).__init__()
        self.encoder = encoder
        #""" Think About Duplicating Decoders Here if You Want (Maybe in the Form of a Module Dict) """
        self.decoder = decoder
    
    def forward(self,x,text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device):
        """ Encode Frame """
        h, annot_vectors, class_predictions = self.encoder(x)
        """ Check Dimension of h """
        
        """ Feed h as Initialization of Hidden Vectors of LSTM """ #sorted_indices, attn_coefs
        outputs_padded = self.decoder(text_indices,sentence_lens,token2id_dict,id2embedding_dict,dest_lang,phase,sampling,device,h0=h,c0=h,annot_vectors=annot_vectors)

        return outputs_padded, h #sorted_indices, attn_coefs, class_predictions

#%%
#""" Use These Networks for the Visual Question Answering Task """    
#    
#class text_encoder_network(nn.Module):
#    
#    def __init__(self,embedding_dim):
#        super(text_encoder_network,self).__init__()
#        input_features = embedding_dim 
#        hidden_features = embedding_dim
#        num_layers = 1
#        lstm = nn.LSTM(input_features,hidden_features,num_layers,batch_first=True)
#        
#        self.lstm = lstm
#        self.num_layers = num_layers
#        self.hidden_features = hidden_features
#        
#    def forward(self,text_indices,sentence_lens,id2embedding_dict,phase,device): #remember, this is at the batch level
#        """ Return Final Decoder State """
#        batch_size = len(text_indices)
#        h0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device) #1xBxL
#        c0 = torch.randn(self.num_layers,batch_size,self.hidden_features,device=device)
#
#        """ Convert Text Indices to List """
#        text_indices = text_indices.tolist()
#        """ Convert Padded Sequences to Embeddings """
#        text = torch.stack([torch.stack(itemgetter(*text_index)(id2embedding_dict)) for text_index in text_indices],0) #BxSxL
#        
#        """ Pack Sequences (CHECK THE ENFORCE SORTED CONDITION) """
#        text_packed = pack_padded_sequence(text,sentence_lens,batch_first=True,enforce_sorted=False)
#        """ Pass Them Through LSTM (One by One) """
#        outputs_packed, (hn, cn) = self.lstm(text_packed,(h0,c0))
#        outputs_padded, outputs_lengths = pad_packed_sequence(outputs_packed, batch_first=True)
#        """ Detach Initial Hidden States to Avoid Memory Leakage """
#        h0 = h0.detach()
#        c0 = c0.detach()
#        #hn is 1xBxL
#        return hn
#
class vqa_decoder_network(nn.Module):
    
    def __init__(self,embedding_dim,attn=False):
        super(vqa_decoder_network,self).__init__()
        input_features = embedding_dim * 2 #if attn is not None else embedding_dim
        hidden_features = embedding_dim
        output_features = 101 # 1 for regression
        """ Prediction Layers """
        linear1 = nn.Linear(input_features,hidden_features)
        linear2 = nn.Linear(hidden_features,output_features)
        
        #for name,param in linear2.named_parameters():
        #    if 'bias' in name:
        #        param.data.copy_(torch.tensor(100))
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = linear1
        self.linear2 = linear2
        self.relu = nn.ReLU()
        self.attn = attn

        if self.attn == 'concat_nonlinear':
            """ Attention Layers """
            self.linear_attn1a = nn.Linear(input_features,hidden_features)
            self.linear_attn1b = nn.Linear(input_features,hidden_features)
            self.linear_attn2 = nn.Linear(hidden_features,1)
            self.Tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()

    def calculate_attn_coefs(self,text_vector,annot_vectors,image_regions):
        """ Calculate Attention Coefficients When in the Training Stage (i.e. Batched) """
        if self.attn is not None:
            if self.attn == 'direct_dot_product':
                #""" Normalize All Vectors (L2 Norm) """
                #annot_vectors_norm = annot_vectors.norm(dim=1).unsqueeze(1).repeat(1,annot_vectors.shape[1],1) # Bx128x79
                #text_vector_norm = text_vector.norm(dim=2).unsqueeze(2).repeat(1,1,text_vector.shape[2]) # 1xBxL
                #annot_vectors_normed = annot_vectors/annot_vectors_norm
                #text_vector_normed = text_vector/text_vector_norm
                """ Calculate Attention Coefficients """
                attn_coefs = torch.bmm(annot_vectors.permute(0,2,1),text_vector.permute(1,2,0)) # Bx79x128  X  Bx128x1 = Bx79x1   
                attn_coefs = F.softmax(attn_coefs,dim=1)
            elif self.attn == 'concat_nonlinear':
                modified_text = text_vector.permute(1,2,0) #BxLx1
                modified_text = modified_text.repeat(1,1,image_regions).permute(0,2,1) #Bx79xL
                modified_annot_vectors = annot_vectors.permute(0,2,1) #Bx79xL
                #modified_annot_vectors = modified_annot_vectors.repeat(1,text.shape[1],1,1).permute(0,1,3,2) #BxSx79xL 
                """ Concatenate Two Vectors """
                concat_vectors = torch.cat((modified_text,modified_annot_vectors),2) #Bx79x2L 
                """ Calculate Attention """
                intermediate_coefs_a = self.Tanh(self.linear_attn1a(concat_vectors)) #Bx79xL
                intermediate_coefs_b = self.sigmoid(self.linear_attn1b(concat_vectors)) #Bx79xL
                intermediate_coefs = intermediate_coefs_a * intermediate_coefs_b
                attn_coefs = self.linear_attn2(intermediate_coefs) #Bx79x1
                attn_coefs = F.softmax(attn_coefs,dim=1) 
            
            context_vectors = torch.bmm(annot_vectors,attn_coefs) # Bx128x79  X Bx79x1 = Bx128x1 
            combined_encoder_vector = torch.cat((text_vector.permute(1,0,2),context_vectors.permute(0,2,1)),2).squeeze(1) # Bx(2*L)         
        elif self.attn == None:
            context_vectors = torch.mean(annot_vectors,-1).unsqueeze(-1).permute(0,2,1) # Bx1xL
            #print(context_vectors.shape,text_vector.shape)
            #print(context_vectors,text_vector)
            combined_encoder_vector = torch.cat((text_vector.permute(1,0,2),context_vectors),2).squeeze(1) # Bx(2*L)         
            attn_coefs = torch.rand(2) #filler
        
        return combined_encoder_vector, attn_coefs

    def forward(self,annot_vectors,text_vector):
        """ Make Prediction/Answer the Question 
        Args:
            annot_vectors (torch.Tensor): BxLx79
            text_vector (torch.Tensor): 1xBxL
        Returns:
            answer (torch.Tensor): scalar
        """
        
        image_regions = annot_vectors.shape[-1]
        """ Obtain Combined Encoded Vector """
        combined_encoder_vector, attn_coefs = self.calculate_attn_coefs(text_vector,annot_vectors,image_regions)
        """ Obtain Answers """
        h = self.relu(self.linear1(combined_encoder_vector))
        answers = self.linear2(h)
        #print(answers.shape)
#        combined_encoder_vector = self.avgpool(annot_vectors).squeeze()
#        """ Obtain Answers """
#        h = self.relu(self.linear1(combined_encoder_vector))
#        answers = self.linear2(h)
#        attn_coefs = answers
        
        return answers, attn_coefs

class combined_vqa_network(nn.Module):
    
    def __init__(self,image_encoder,text_encoder,vqa_decoder):
        super(combined_vqa_network,self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.vqa_decoder = vqa_decoder
    
    def forward(self,frame,text_indices,sentence_lens,id2embedding_dict,phase,device):
        """ Encode Frame """
        h, annot_vectors, class_predictions = self.image_encoder(frame)
        #for param in self.image_encoder.parameters():
        #    print(param.requires_grad)

        """ Encode Text """
        text_vector = self.text_encoder(text_indices,sentence_lens,id2embedding_dict,phase,device)
        """ Combined Visual and Linguistic Vectors and Decode """
        answers, attn_coefs = self.vqa_decoder(annot_vectors,text_vector)
        
        return answers, h, attn_coefs, class_predictions
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
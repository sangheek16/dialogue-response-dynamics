import itertools
import random
import sys
import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer
import datetime
from minicons import scorer
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch

# header selection file 
# --- config --- #
INPUT_PAIRS = '../datasets/used_items.csv'
OUTPUT_DIR = '../out'
model_name= "roberta-base" # supported models ['distilbert-base-cased', 'distilroberta-base', 'xlm-roberta-base', 'distilgpt2', 'roberta-base', 'bert-base-cased', 'gpt2']
task = 'header_selection' # available tasks: [header_selection, rejection, conjunction, ellipsis]


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


if model_name == 'distilgpt2':
    model_type = 'CLM'
else:
    model_type = 'MLM'

_start_ind = 0
input_size = 600 #1200 # length of the input dataframe
chunk_size = input_size - _start_ind #<-- needs to be in multiples of 100

'''
--------------------------------------------------------------------
    Input Prep
--------------------------------------------------------------------
'''

# cleaning input material
df = pd.read_csv(INPUT_PAIRS)
df1, df2 = df.copy(deep=True), df.copy(deep=True)
df1['header'], df2['header'] = 'reject', 'wait'
df = pd.concat([df1, df2])
df.reset_index(inplace=True, drop=True)

# dictionary for header
header = {}
header['reject'] = 'No'; header['wait'] = 'Wait no'


'''
--------------------------------------------------------------------
    Helper Functions
--------------------------------------------------------------------
'''

def fill_mask_target(model_name, seq, target_words, tokenizer, lang_model):
    # target_words should be in list

    sequence = seq.replace('[MASK]', tokenizer.mask_token)
    inputs = tokenizer(sequence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits = lang_model(**inputs).logits 
    mask_token_logits = token_logits[0, mask_token_index, :]
    probs = mask_token_logits.softmax(dim = 1)
    
    target_ids = dict()
    for word in target_words:
        if 'roberta' in model_name:
            target_ids[word] = tokenizer.encode(str(' ') + word)[1]
        else:
            target_ids[word] = tokenizer.encode(word)[1]

    target_probs = dict()
    for k,v in target_ids.items():
            target_probs[k] = round(probs[..., v].item(),4)
    return target_probs

'''
--------------------------------------------------------------------
    Task Logic
--------------------------------------------------------------------
'''

def header_selection(df, model_type):
    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)
    if model_type == 'CLM':

        clm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')

        def scorer_clm(x):
            start_time = datetime.datetime.now()
            score_raw = clm_model.sequence_score(x)[0] # log probability (/ surprisal) normalized by number of tokens
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1
            print('It took %d seconds to process this row' %(execution_time))
            return score_raw

        data_clm = df.copy(deep=True)
        data_clm['model_name'] = 'distilgpt2'
        data_clm['score1'] = data_clm['seq1'].map(lambda x: scorer_clm(x))
        data_clm['score2'] = data_clm['seq2'].map(lambda x: scorer_clm(x))

        data_clm.to_csv('%s/%s_header_selection.csv' %(OUTPUT_DIR, model_name), index=False)
    elif model_type == 'MLM':

        if model_name == 'distilbert-base-cased':
            print('> .. Importing distilbert-base-cased')
            mlm_model = scorer.MaskedLMScorer('distilbert-base-cased', 'cpu')
        elif model_name == 'distilroberta-base':
            print('> .. Importing distilroberta-base')
            mlm_model = scorer.MaskedLMScorer('distilroberta-base', 'cpu')
        elif model_name == 'xlm-roberta-base':
            print('> .. Importing xlm-roberta-base')
            mlm_model = scorer.MaskedLMScorer('xlm-roberta-base', 'cpu')
        elif model_name == 'roberta-base':
            print('> .. Importing roberta-base')
            mlm_model = scorer.MaskedLMScorer('roberta-base', 'cpu')
        elif model_name == 'bert-base-cased':
            print('> .. Importing bert-base-cased')
            mlm_model = scorer.MaskedLMScorer('bert-base-cased', 'cpu')

        def scorer_mlm(x):
            start_time = datetime.datetime.now()
            score_raw = mlm_model.sequence_score(x)[0] # log probability normalized by number of tokens
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1
            print('It took %d seconds to process this row' %(execution_time))
            return score_raw

        data_mlm = df.copy(deep=True)
        data_mlm['model_name'] = model_name
        data_mlm['score1'] = data_mlm['seq1'].map(lambda x: scorer_mlm(x))
        data_mlm['score2'] = data_mlm['seq2'].map(lambda x: scorer_mlm(x))

        data_mlm.to_csv('%s/%s_header_selection.csv' %(OUTPUT_DIR,  model_name), index=False)
    else:
        print("unsupported model type {}".format(model_type))


def rejection(df):
    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    prediction_dict = {}
    row_count = 0
    total_time = 0

    print('** Getting predictions ... **')
    for idn, row in df.iloc[:_start_ind + chunk_size].iterrows():
        if idn < _start_ind:
            row_count += 1
            continue

        start = timer()

        verb1, verb2 = row['verb1'], row['verb2']
        vp1, vp2 = row['vp1'], row['vp2']
        subj, prn, name1, name2 = row['subj'], row['prn'], row['name1'], row['name2']
        header = row['header']
        seq1, seq2 = row['seq1'], row['seq2']

        prediction_dict[row_count] = {}
        prediction_dict[row_count]['header'] = header
        prediction_dict[row_count]['subj'] = subj
        prediction_dict[row_count]['prn'] = prn
        prediction_dict[row_count]['name1'] = name1
        prediction_dict[row_count]['name2'] = name2
        prediction_dict[row_count]['verb1'] = verb1
        prediction_dict[row_count]['verb2'] = verb2
        prediction_dict[row_count]['vp1'] = vp1
        prediction_dict[row_count]['vp2'] = vp2
        prediction_dict[row_count]['model'] = model_name

        seq1 = seq1.replace(verb1 + ' not.', '[MASK] not.')
        seq2 = seq2.replace(verb2 + ' not.', '[MASK] not.')
        prediction_dict[row_count]['seq1'] = seq1
        prediction_dict[row_count]['seq2'] = seq2
        prediction_dict[row_count]['score1'] = fill_mask_target(model_name, seq1, [verb1], tokenizer, lang_model)[verb1]
        prediction_dict[row_count]['score2'] = fill_mask_target(model_name, seq2, [verb2], tokenizer, lang_model)[verb2]

        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size)) # print('%d/%d' %(row_count, len(df)*len(_models_clm+_models_mlm)))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'rejection-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)


def conjunction(df):
    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s %s and %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s %s and %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    prediction_dict = {}
    row_count = 0
    total_time = 0

    print('** Getting predictions ... **')
    for idn, row in df.iloc[:_start_ind + chunk_size].iterrows():
        if idn < _start_ind:
            row_count += 1
            continue

        start = timer()

        verb1, verb2 = row['verb1'], row['verb2']
        vp1, vp2 = row['vp1'], row['vp2']
        subj, prn, name1, name2 = row['subj'], row['prn'], row['name1'], row['name2']
        header = row['header']
        seq1, seq2 = row['seq1'], row['seq2']

        prediction_dict[row_count] = {}
        prediction_dict[row_count]['header'] = header
        prediction_dict[row_count]['subj'] = subj
        prediction_dict[row_count]['prn'] = prn
        prediction_dict[row_count]['name1'] = name1
        prediction_dict[row_count]['name2'] = name2
        prediction_dict[row_count]['verb1'] = verb1
        prediction_dict[row_count]['verb2'] = verb2
        prediction_dict[row_count]['vp1'] = vp1
        prediction_dict[row_count]['vp2'] = vp2
        prediction_dict[row_count]['model'] = model_name

        seq1 = seq1.replace(verb1 + ' not.', '[MASK] not.')
        seq2 = seq2.replace(verb2 + ' not.', '[MASK] not.')
        prediction_dict[row_count]['seq1'] = seq1
        prediction_dict[row_count]['seq2'] = seq2
        prediction_dict[row_count]['value1'] = fill_mask_target(model_name, seq1, [verb1], tokenizer, lang_model)[verb1]
        prediction_dict[row_count]['value2'] = fill_mask_target(model_name, seq2, [verb2], tokenizer, lang_model)[verb2]

        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size)) # print('%d/%d' %(row_count, len(df)*len(_models_clm+_models_mlm)))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'conjunction-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)


def ellipsis(df):
    # create input sequence
    df['seq'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s [VERB] not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn']), axis=1)

    verb_list = list(df.verb1.unique())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    prediction_dict = {}
    row_count = 0
    total_time = 0


    print('** Getting predictions ... **')
    for idn, row in df.iloc[:_start_ind + chunk_size].iterrows():

        if idn < _start_ind:
            row_count += 1
            continue

        start = timer()
        verb1, verb2 = row['verb1'], row['verb2']
        vp1, vp2 = row['vp1'], row['vp2']
        subj, prn, name1, name2 = row['subj'], row['prn'], row['name1'], row['name2']
        header = row['header']
        seq = row['seq']

        prediction_dict[row_count] = {}
        prediction_dict[row_count]['header'] = header
        prediction_dict[row_count]['subj'] = subj
        prediction_dict[row_count]['prn'] = prn
        prediction_dict[row_count]['name1'] = name1
        prediction_dict[row_count]['name2'] = name2
        prediction_dict[row_count]['verb1'] = verb1
        prediction_dict[row_count]['verb2'] = verb2
        prediction_dict[row_count]['vp1'] = vp1
        prediction_dict[row_count]['vp2'] = vp2

        prediction_dict[row_count]['seq'] = seq
        prediction_dict[row_count]['vs_pair'] = fill_mask_target(model_name, seq, verb_list, tokenizer, lang_model)

        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size)) # print('%d/%d' %(row_count, len(df)*len(_models_clm+_models_mlm)))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'ellipsis-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)


if task == 'header_selection':
    header_selection(df, model_type)
elif task == 'rejection':
    rejection(df)
elif task == 'conjunction':
    conjunction(df)
elif task == 'ellipsis':
    ellipsis(df)
else:
    print("unsupported task {}".format(task))
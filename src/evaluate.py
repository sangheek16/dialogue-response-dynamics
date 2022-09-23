import pandas as pd
import os
from timeit import default_timer as timer
import datetime
from minicons import scorer
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch


# --- config --- #
INPUT_PAIRS = '../datasets/used_items.csv'
OUTPUT_DIR = '../out'

# create path
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# supported models and tasks
_supported_models = ['bert-base-cased', 'roberta-base', 'xlm-roberta-base', 'distilbert-base-cased', 'distilroberta-base', 'distilgpt2']
_supported_tasks = ['header_selection', 'rejection', 'conjunction', 'ellipsis']


'''
--------------------------------------------------------------------
    Input Prep
--------------------------------------------------------------------
'''

# including headers in the input
df_input = pd.read_csv(INPUT_PAIRS)
df_input1, df_input2 = df_input.copy(deep=True), df_input.copy(deep=True)
df_input1['header'], df_input2['header'] = 'reject', 'wait'
df_input = pd.concat([df_input1, df_input2])
df_input.reset_index(inplace=True, drop=True)


'''
--------------------------------------------------------------------
    Helper Functions
--------------------------------------------------------------------
'''

# function for getting prediction with clm
def scorer_clm(x, clm_model):
    start_time = datetime.datetime.now()
    score_raw = clm_model.sequence_score(x)[0] # log probability (/ surprisal) normalized by number of tokens
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1
    print('It took %d seconds to process this row' %(execution_time))
    return score_raw

# function for getting prediction with mlm
# .. for rejection & conjunction tasks
def fill_mask_target(model_name, seq, target_words, tokenizer, lang_model):
    # target_words should be in a list (e.g., ['did'], ['did', 'does'])

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
    return target_probs # different output from fill_verb_target

# function for getting prediction with mlm
# .. for ellipsis task
def fill_verb_target(model_name, seq, target_words, tokenizer, lang_model):
    # target_words should be in list

    sequence = seq.replace('[VERB]', tokenizer.mask_token)
    inputs = tokenizer(sequence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits = lang_model(**inputs).logits 
    mask_token_logits = token_logits[0, mask_token_index, :]
    probs = mask_token_logits.softmax(dim = 1)
    
    target_ids = dict()
    for word in target_words:
        if model_name == "distilroberta-base":
                target_ids[word] = tokenizer.encode(str(' ') + word)[1]
        else:
                target_ids[word] = tokenizer.encode(word)[1]

    target_probs = dict()
    for k,v in target_ids.items():
            target_probs[k] = round(probs[..., v].item(),4)
    return list(target_probs.items()) # different output from fill_mask_target

'''
--------------------------------------------------------------------
    Task Logic
--------------------------------------------------------------------
'''

def header_selection(df, model_type):
    # dictionary for header
    header = {}
    header['reject'] = 'No'; header['wait'] = 'Wait no'

    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)
    
    # get predictions
    print('** Getting predictions ... **')

    if model_type == 'CLM':
        clm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')

        df['model'] = model_name
        df['score1'] = df['seq1'].map(lambda x: scorer_clm(x, clm_model))
        df['score2'] = df['seq2'].map(lambda x: scorer_clm(x, clm_model))

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

        df['model'] = model_name
        df['score1'] = df['seq1'].map(lambda x: scorer_mlm(x))
        df['score2'] = df['seq2'].map(lambda x: scorer_mlm(x))

    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
    if not os.path.exists(_output_folder):
        os.makedirs(_output_folder)
    df.to_csv(_output_folder + 'header_selection-%s.csv' %(model_name), index=False)
    # return data_mlm


def rejection(df, model_type):
    # dictionary for header
    header = {}
    header['reject'] = 'No'; header['wait'] = 'Wait no'

    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)

    # initialize language models
    if model_type == 'CLM':
        clm_model = scorer.IncrementalLMScorer(model_name, 'cpu')

    elif model_type == 'MLM':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    # get predictions
    print('** Getting predictions ... **')

    prediction_dict = {}
    row_count = 0
    total_time = 0

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

        if model_type == 'CLM':
            prediction_dict[row_count]['seq1'] = seq1
            prediction_dict[row_count]['seq2'] = seq2
            prediction_dict[row_count]['score1'] = scorer_clm(seq1, clm_model)
            prediction_dict[row_count]['score2'] = scorer_clm(seq2, clm_model)     

        elif model_type == 'MLM':
            prediction_dict[row_count]['seq1'] = seq1
            prediction_dict[row_count]['seq2'] = seq2
            seq1 = seq1.replace(verb1 + ' not.', '[MASK] not.')
            seq2 = seq2.replace(verb2 + ' not.', '[MASK] not.')
            prediction_dict[row_count]['score1'] = fill_mask_target(model_name, seq1, [verb1], tokenizer, lang_model)[verb1]
            prediction_dict[row_count]['score2'] = fill_mask_target(model_name, seq2, [verb2], tokenizer, lang_model)[verb2]

        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'rejection-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)
    # return prediction_dict


def conjunction(df, model_type):
    # dictionary for header
    header = {}
    header['reject'] = 'No'; header['wait'] = 'Wait no'

    # create input sequence
    df['seq1'] = df.apply(lambda x: '%s said, "%s %s and %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb1']), axis=1)
    df['seq2'] = df.apply(lambda x: '%s said, "%s %s and %s," and %s replied, "%s, %s %s not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn'], x['verb2']), axis=1)

    # initialize language models
    if model_type == 'CLM':
        clm_model = scorer.IncrementalLMScorer(model_name, 'cpu')

    elif model_type == 'MLM':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    # get predictions
    print('** Getting predictions ... **')

    prediction_dict = {}
    row_count = 0
    total_time = 0

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

        if model_type == 'CLM':
            prediction_dict[row_count]['seq1'] = seq1
            prediction_dict[row_count]['seq2'] = seq2
            prediction_dict[row_count]['score1'] = scorer_clm(seq1, clm_model)
            prediction_dict[row_count]['score2'] = scorer_clm(seq2, clm_model)

        elif model_type == 'MLM':
            prediction_dict[row_count]['seq1'] = seq1
            prediction_dict[row_count]['seq2'] = seq2
            seq1 = seq1.replace(verb1 + ' not.', '[MASK] not.')
            seq2 = seq2.replace(verb2 + ' not.', '[MASK] not.')
            prediction_dict[row_count]['score1'] = fill_mask_target(model_name, seq1, [verb1], tokenizer, lang_model)[verb1]
            prediction_dict[row_count]['score2'] = fill_mask_target(model_name, seq2, [verb2], tokenizer, lang_model)[verb2]
            
        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'conjunction-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)

    # return prediction_dict


def ellipsis(df, model_type):
    # dictionary for header
    header = {}
    header['reject'] = 'No'; header['wait'] = 'Wait no'

    # create input sequence
    df['seq'] = df.apply(lambda x: '%s said, "%s, who %s, %s," and %s replied, "%s, %s [VERB] not."' %(x['name1'], x['subj'], x['vp1'], x['vp2'], x['name2'], header[x['header']], x['prn']), axis=1)

    verb_list = list(df.verb1.unique()) # verb_list = list(df.verb2.uniuqe())

    # initialize language models
    if model_type == 'CLM':
        clm_model = scorer.IncrementalLMScorer(model_name, 'cpu')

    elif model_type == 'MLM':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lang_model = AutoModelForMaskedLM.from_pretrained(model_name)

    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    # get predictions
    print('** Getting predictions ... **')

    prediction_dict = {}
    row_count = 0
    total_time = 0

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

        if model_type == 'CLM':
            prediction_dict[row_count]['vs_pair'] = []
            for verb in verb_list:
                seq_replaced = seq.replace('[VERB]', verb)
                verb_score = (verb, scorer_clm(seq_replaced, clm_model))
                prediction_dict[row_count]['vs_pair'].append(verb_score)
        
        elif model_type == 'MLM':
            prediction_dict[row_count]['vs_pair'] = fill_verb_target(model_name, seq, verb_list, tokenizer, lang_model)

        end = timer()
        row_count += 1

        print('%d/%d' %(row_count, chunk_size))
        print('elapsed time for one iteration: %d' %(end-start))
        total_time += end-start
        print('total elapsed time: %d' %(total_time))

        if row_count % 100 == 0:
            _output_folder = '%s/%s/' %(OUTPUT_DIR, model_name)
            if not os.path.exists(_output_folder):
                os.makedirs(_output_folder)
            pd.DataFrame(prediction_dict).T.to_csv(_output_folder + 'ellipsis-evaluation-%s_row%d.csv' %(model_name, row_count), index=False)
    # return prediction_dict

'''
--------------------------------------------------------------------
    Run Tasks
--------------------------------------------------------------------
'''

# run by chunk
_start_ind = 0
input_size = 600 # maximum input_size is 600 (= 6P2 verb pair combinations * 10 items * 2 headers)
chunk_size = input_size - _start_ind # needs to be a multiple of 100

# --- assign model
model_name= 'distilgpt2' # an example # _supported_models[5]

if model_name not in _supported_models:
    raise ValueError("Unsupported model: {}".format(model_name))
elif model_name == 'distilgpt2':
    model_type = 'CLM'
else:
    model_type = 'MLM'
print("** Model: {} / Model type: {}".format(model_name, model_type))

# --- assign task
task = 'header_selection' # an example # _supported_tasks[0]

if task not in _supported_tasks:
    raise ValueError("Unsupported task: {}".format(task))

elif task == 'header_selection':
    print("** Task: {}".format(task))
    header_selection(df_input, model_type)

elif task == 'rejection':
    print("** Task: {}".format(task))
    rejection(df_input, model_type)

elif task == 'conjunction':
    print("** Task: {}".format(task))
    conjunction(df_input, model_type)

elif task == 'ellipsis':
    print("** Task: {}".format(task))
    ellipsis(df_input, model_type)

print("** -- evaluate.py completed -- **")
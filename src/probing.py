import pandas as pd
from regex import search

from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import random

# --- config --- #
INPUT_PAIRS = '../datasets/used_items.csv'
model_name = "xlm-roberta-base"

print("init model...")

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)

_start_ind = 0
input_size = 600 # length of the input dataframe
chunk_size = input_size - _start_ind #<-- needs to be in multiples of 100

print('Model: %s / From %d - %d' %(model_name, _start_ind, _start_ind + chunk_size))

df = pd.read_csv(INPUT_PAIRS)
df['name1'] = df['arc_temp'].map(lambda x : x.split('said')[0].strip(' '))
df['name2'] = df['arc_temp'].map(lambda x : x.split('replied')[0].strip(' ').split(' ')[-1])
df1, df2 = df.copy(deep=True), df.copy(deep=True)
df1['header'], df2['header'] = 'reject', 'wait'
df = pd.concat([df1, df2])
df.reset_index(inplace=True, drop=True)

verb_list = list(df.verb1.unique())

def search_target_range(input_ids, search_ids):
    search_targets = search_ids[1:-1]  # removing cls and sep
    size = len(search_targets)
    for i in range(len(input_ids)):
        if input_ids[i: i + size] == search_targets:
            return i, i + size

    print("can't find target vp!")
    exit(1)

clf_embeddings_train = []
clf_labels_train = []
clf_embeddings_test = []
clf_labels_test = []

for ind, row in df.iloc[:_start_ind + chunk_size].iterrows():
    in_sequence = row['arc']
    at_issue = row['vp2']
    not_at_issue = row['vp1']

    inputs = tokenizer(in_sequence, return_tensors="pt")
    out = model(**inputs)
    hidden_states = out.last_hidden_state  # (1, seq_len, hidden_size)

    at_issue_ids = tokenizer.encode(str(' ') + at_issue)
    not_at_issue_ids = tokenizer.encode(str(' ') + not_at_issue)

    input_ids = tokenizer.encode(in_sequence)
    
    at_issue_lower, at_issue_upper = search_target_range(input_ids, at_issue_ids)
    not_at_issue_lower, not_at_issue_upper = search_target_range(input_ids, not_at_issue_ids)

    if random.random() <= 0.7:

        for i in range(hidden_states.shape[1]):
            clf_embeddings_train.append(hidden_states[0][i].detach().numpy())
            if i >= at_issue_lower and i < at_issue_upper:
                clf_labels_train.append(1)
            elif i >= not_at_issue_lower and i < not_at_issue_upper:
                clf_labels_train.append(0)
            else:
                clf_labels_train.append(2)
    else:
        for i in range(hidden_states.shape[1]):
            clf_embeddings_test.append(hidden_states[0][i].detach().numpy())
            if i >= at_issue_lower and i < at_issue_upper:
                clf_labels_test.append(1)
            elif i >= not_at_issue_lower and i < not_at_issue_upper:
                clf_labels_test.append(0)
            else:
                clf_labels_test.append(2)

    
print("training size: {}, test size: {}".format(len(clf_embeddings_train), len(clf_embeddings_test)))

clf = MLPClassifier(hidden_layer_sizes=50, activation="relu", max_iter=300)
clf.fit(np.asarray(clf_embeddings_train), np.asarray(clf_labels_train))

clf_pred = clf.predict(clf_embeddings_test)
clf_acc = accuracy_score(clf_pred, clf_labels_test)

print("classification accuracy: {}".format(clf_acc))
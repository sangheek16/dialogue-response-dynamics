import itertools
import pandas as pd
import random

_num_samples = 10

'-------------------------------------'
# read material
df_np = pd.read_csv('../datasets/np.csv')
df_np.applymap(lambda x: x.strip())

df_vp = pd.read_csv('../datasets/vp.csv')
df_vp.applymap(lambda x: x.strip())

df_name = pd.read_csv('../datasets/names.csv')
df_name.applymap(lambda x: x.strip())
ls_name = df_name['names'].tolist()

# noun phrase and pronoun
np_dict = dict(zip(df_np.np, df_np.prn))

# verb phrase & verb
vp_dict = dict(zip(df_vp.vp, df_vp.vp_res))

# verb combination 
verb_dict = {}
for verb in list(df_vp['vp_res'].unique()):
    verb_dict[verb] = list(df_vp.loc[df_vp['vp_res'] == verb].vp.unique())

# get 6 permutation of 2 verb pairs
verb_list = list(df_vp['vp_res'].unique())
verb_permutation = tuple(itertools.permutations(verb_list, 2))
print('** the number of permutations: %d **' %(len(verb_permutation)))

verb_tuples = {}
for verb_tuple in verb_permutation:
    verb_tuples[verb_tuple] = []
    num_tuple = 0
    while num_tuple < _num_samples:
        verb1, verb2 = verb_tuple[0], verb_tuple[1]
        vp1 = random.choice(verb_dict[verb1])
        vp2 = random.choice(verb_dict[verb2])
        vp_tuple = (vp1, vp2)
        if vp_tuple in verb_tuples[verb_tuple]:
            pass
        else: 
            verb_tuples[verb_tuple].append(vp_tuple)
            num_tuple += 1

print('** the length of values for each key should be %d **' %(_num_samples))
print({k: len(v) for k, v in verb_tuples.items()})

# sequence generation
df_items = pd.DataFrame(columns = ['verb1','verb2','vp1','vp2',
                                      'subj','prn', 'name1', 'name2'])

row_count = 0
for k, v in verb_tuples.items():
    for vp_tuple in v:
        verb1, verb2 = k[0], k[1]
        vp1, vp2 = vp_tuple[0], vp_tuple[1]
        np = random.choice(df_np['np'].tolist())
        prn = np_dict[np]
        names = random.sample(ls_name,2)
        name1, name2 = names[0], names[1]

        df_items.loc[row_count, 'verb1'] = verb1
        df_items.loc[row_count, 'verb2'] = verb2
        df_items.loc[row_count, 'vp1'] = vp1
        df_items.loc[row_count, 'vp2'] = vp2
        df_items.loc[row_count, 'subj'] = np
        df_items.loc[row_count, 'prn'] = prn        
        df_items.loc[row_count, 'name1'] = name1
        df_items.loc[row_count, 'name2'] = name2

        row_count += 1

print('** size should be: %d; output size: %d' %(_num_samples * len(verb_permutation), len(df_items)))
df_items.to_csv('../datasets/new_items-nsample=%s.csv' %(_num_samples), index=False)

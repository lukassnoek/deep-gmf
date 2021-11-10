from genericpath import isfile
import os
import os.path as op
import pandas as pd
from glob import glob

dset_path = op.join('data', 'human_exp')
info = {'file': [], 'face_id': [], 'sex': []}
# Recode into proper strings (otherwise to_categorical doesn't work)
id2str = {'501': 'f1', '502': 'f2', '503': 'f3', '504': 'f4'}
for root, dirs, files in os.walk(dset_path):
    
    for f in files:
        if f.split('.')[-1] == 'png':
            f_path = os.path.join(root, f)
            info['file'].append(f_path)
            f_split = f.split('_')
            info['face_id'].append(id2str[f_split[0]])
            info['sex'].append(f_split[1][1])
            
df = pd.DataFrame(info)
df.to_csv(op.join(dset_path, 'dataset_info.csv'), index=False)
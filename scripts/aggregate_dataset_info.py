import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path('/analyse/Project0257/lukas/data/mini_gmf_dataset')
MAPPING = {
    'xr': {'0': -45, '1': 0, '2': 45},
    'yr': {'0': -45, '1': 0, '2': 45},
    'zr': {'0': -45, '1': 0, '2': 45},
    'xt': {'0': -150, '1': 0, '2': 150},   
    'yt': {'0': -150, '1': 0, '2': 150},
    'l': {'0': 'front', '1': 'above', '2': 'below',
          '3': 'left', '4': 'right'}
}

files = DATA_DIR.glob('**/*.png')
info = defaultdict(list)
for f in tqdm(files):
    info['filename'].append(f)
    f = str(f).split('/analyse/Project0257/lukas/data/mini_gmf_dataset/')[1]
    for subdir in f.split('/'):
        if not '_' in subdir:
            parts = [subdir]
        else:
            parts = subdir.split('_')
        
        for part in parts:
            if '.png' in part:
                part = part[:-4]
                
            k, v = part.split('-')
            if k in MAPPING.keys():
                v = MAPPING[k][v]

            info[k].append(v) 

df = pd.DataFrame.from_dict(info)
df.to_csv(DATA_DIR / 'dataset_info.csv', index=False)
print(df)
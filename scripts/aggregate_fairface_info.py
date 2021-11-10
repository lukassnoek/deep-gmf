import pandas as pd

dfs = []
for split in ('train', 'val'):
    
    f = f'data/fairface-img-margin025-trainval/fairface_label_{split}.csv'
    df = pd.read_csv(f)
    ages = []
    for age in df['age']:
        if 'more' in age:
            ages.append(75)
        else:
            ages.append(int(sum(int(a) for a in age.split('-')) // 2))
            
    df = df.assign(age=ages)
    df['split'] = split
    df['file'] = 'data/fairface-img-margin025-trainval/' + df['file']
    dfs.append(df)    

df = pd.concat(dfs, axis=0)
df.to_csv('data/fairface-img-margin025-trainval/dataset_info.csv', index=False)

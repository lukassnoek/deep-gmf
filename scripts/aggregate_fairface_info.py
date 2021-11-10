import pandas as pd


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
    df.to_csv(f.replace('.csv', '_fixed.csv'), index=False)

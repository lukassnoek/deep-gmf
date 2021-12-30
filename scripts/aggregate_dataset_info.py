import click
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

MAPPING = {
    'xr': {'0': -45, '1': 0, '2': 45},
    'yr': {'0': -45, '1': 0, '2': 45},
    'zr': {'0': -45, '1': 0, '2': 45},
    'xt': {'0': -150, '1': 0, '2': 150},   
    'yt': {'0': -150, '1': 0, '2': 150},
    'l': {'0': '0front', '1': '1above', '2': '2below',
        '3': '3left', '4': '4right'}
}

@click.command()
@click.argument('dataset_name')
def main(dataset_name):

    data_dir = Path(f'/analyse/Project0257/lukas/data/{dataset_name}')
    files = data_dir.glob('**/*.png')
    info = defaultdict(list)
    for f in tqdm(files):
        info['filename'].append(f)
        f = str(f).split(f'{dataset_name}/')[1]
        parts = f.replace('/', '_').replace('.png', '').split('_')
        for part in parts:    
            k, v = part.split('-')
            if k in MAPPING.keys():
                v = MAPPING[k][v]

            info[k].append(v) 

    df = pd.DataFrame.from_dict(info)
    df = df.sort_values(by=['id', 'gender', 'ethn', 'age', 'bg',
                            'xr', 'yr', 'zr', 'xt', 'yt', 'l'], axis=0)
    #print(df)
    df.to_csv(data_dir / 'dataset_info.csv', index=False)
    corrs = pd.get_dummies(df.drop('filename', axis=1)).corr()
    print(corrs.round(3))


if __name__ == '__main__':
    
    main()
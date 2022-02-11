import click
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import product, chain
from random import sample


@click.command()
@click.argument('dataset_name')
@click.option('--n-id-train', type=click.INT, default=48)
@click.option('--n-id-val', type=click.INT, default=24)
@click.option('--n-id-test', type=click.INT, default=24)
def main(dataset_name, n_id_train, n_id_val, n_id_test):
    """ Explanation of defaults, assuming 96 IDs in total:
    There are 12 unique combinations (2 gender x 2 ethn x 2 age),
    and if we want to have at least 2 identities per category,
    the minimum n-id-val/test is (2 x 12 = ) 24
    """
    data_dir = Path(f'/analyse/Project0257/lukas/data/{dataset_name}')
    files = data_dir.glob('**/*image_.png')
    info = defaultdict(list)
    
    for f in tqdm(files):
        info['filename'].append(f)
        f = str(f).split(f'{dataset_name}/')[1]
        parts = f.replace('/', '_').replace('.jpg', '').split('_')
        for part in parts:    
            k, v = part.split('-')
            if k in MAPPING.keys():
                v = MAPPING[k][v]

            info[k].append(v) 
            
    df = pd.DataFrame.from_dict(info)
    df = df.sort_values(by=['id', 'gender', 'ethn', 'age', 'bg',
                            'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'l'], axis=0)

    n_ids = df['id'].nunique()
    if (n_id_train + n_id_val + n_id_test) != n_ids:
        raise ValueError(f"Sum of splits {(n_id_train + n_id_val + n_id_test)} "
                         f"is not the same as number of IDs {n_ids}!")

    # I'm so sorry to the ones reading this for the ugly code! It's randomly drawing
    # identities for the train, val, and test set, making sure that there is an equal
    # proportion of each ethn x gender x age combination (in total: 12)

    df2 = df.copy()  # keep drawing from this DF
    combis = list(product(df['gender'].unique(), df['age'].unique(), df['ethn'].unique()))  # len = 12

    n_per_group = int(n_id_train / len(combis))
    train_ids = [sample(df2.query("(age == @a) & (ethn == @e) & (gender == @g)")['id'].unique().tolist(), n_per_group)
                 for g, a, e in combis]
    train_ids = list(chain(*train_ids))  # flatten nested array
    df.loc[df['id'].isin(train_ids), 'split'] = 'training'  # add to original df
    df2 = df2.query("id not in @train_ids")  # remove from df2 for next sampling set (val)

    n_per_group = int(n_id_val / len(combis))
    val_ids = [sample(df2.query("(age == @a) & (ethn == @e) & (gender == @g)")['id'].unique().tolist(), n_per_group)
                 for g, a, e in combis]
    val_ids = list(chain(*val_ids))
    df.loc[df['id'].isin(val_ids), 'split'] = 'validation'
    df2 = df2.query("id not in @val_ids")

    n_per_group = int(n_id_test / len(combis))
    test_ids = [sample(df2.query("(age == @a) & (ethn == @e) & (gender == @g)")['id'].unique().tolist(), n_per_group)
                 for g, a, e in combis]
    test_ids = list(chain(*test_ids))
    df.loc[df['id'].isin(test_ids), 'split'] = 'testing'
    df2 = df2.query("id not in @test_ids")
    print(df2)  # should be empty if n_train+n_val+n_test = n_id!

    df.to_csv(data_dir / 'dataset_info.csv', index=False)
    #corrs = pd.get_dummies(df.drop('filename', axis=1)).corr()
    #print(corrs.round(3))


if __name__ == '__main__':
    
    main()
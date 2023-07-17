import click
import h5py
import pandas as pd
from random import sample
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


@click.command()
@click.argument('dataset_name')
@click.option('--n-id-train', type=click.INT, default=48)
@click.option('--n-id-val', type=click.INT, default=24)
@click.option('--n-id-test', type=click.INT, default=24)
@click.option('--n-per-id', type=click.INT, default=16)
def main(dataset_name, n_id_train, n_id_val, n_id_test, n_per_id):
    """ Aggregates all ID/stimulus features into a single dataframe
    and splits the data into three 'splits' (training, validation, testing),
    which contain images from unique face IDs.
    
    Parameters
    ----------
    dataset_name : str
        Name of directory with dataset; should be a subdirectory of
        /analyse/Project0257/lukas/data
    n_id_train : int
        Number of face IDs for the train set
    n_id_val : int
        Number of face IDs for validation set
    n_id_test : int
        Number of face IDs for test set
    n_per_id : int
        Number of images per ID; nice for stdout
        
    Raises
    ------
    ValueError
        If the total number of unique face IDs is not the same as
        the sum of the desired train, val, and test IDs
    """

    data_dir = Path(f'/analyse/Project0257/lukas/data/{dataset_name}')

    dirs = sorted(list(data_dir.glob('id-*')))
    info = defaultdict(list)  # keep track of features

    exp_total_files = sum([n_id_train, n_id_val, n_id_test]) * n_per_id    
    i = 0
    for d in tqdm(dirs):

        files = sorted(list(d.glob('*_image.jpg')))
        if len(files) != n_per_id:
            raise ValueError(f"Number of files in {d} is not {n_per_id}!")

        id = d.stem.split('_')[0].split('-')[1]
        if id in info['id']:
            raise ValueError(f"ID {id} already exists in dataframe!")

        for f in files:
            f = str(f)
        
            for ext in ['', 'left', 'right']:
                f_ = f.replace('.jpg', f'{ext}.jpg')
                if not Path(f_).is_file():
                    raise ValueError(f'{f_} does not exist!')
                
                if ext:
                    ext = f'_{ext}'
                
                info[f'image_path{ext}'].append(f)

            feat = f.replace('_image.jpg', '_features.h5')
            info['feature_path'].append(feat)
            
            with h5py.File(feat, mode='r') as f_in:
                
                if id != f_in.attrs['id']:
                    print(f"WARNING: ID in dirname ({id}) does not match ID in {feat} ({f_in.attrs['id']})!")

                # Store all single value features as attributes
                for p in ('xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl', 'emo',
                        'gender', 'ethn', 'age', 'bg', 'id'):
                    
                    if p not in f_in.attrs.keys():
                        print(f"WARNING: attribute {p} does not exist in {feat}!")
                        to_add = 0
                    else:
                        to_add = f_in.attrs[p]

                    info[p].append(to_add)
            
            i += 1

        # Break if we want a subset of all IDs (for quick testing)
        if i > (exp_total_files - 2):
            break

    df = pd.DataFrame.from_dict(info)
    df = df.sort_values(by=['id', 'gender', 'ethn', 'bg', 'age', 'emo', 'xr', 'yr', 
                            'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl'], axis=0)

    # Splits should sum to total number of face IDs
    n_ids = df['id'].nunique()    
    if (n_id_train + n_id_val + n_id_test) != n_ids:
        raise ValueError(f"Sum of splits ({n_id_train + n_id_val + n_id_test}) "
                         f"is not the same as number of IDs {n_ids}!")

    # Split into random train/val/test
    all_ids = df['id'].unique().tolist()
    for n_id, split in [(n_id_train, 'training'), (n_id_val, 'validation'), (n_id_test, 'testing')]:
        these_ids = sample(all_ids, n_id)
        df.loc[df['id'].isin(these_ids), 'split'] = split
        all_ids = [id_ for id_ in all_ids if id_ not in these_ids]

    assert(not all_ids)  # should be empty!
    f_out = str(data_dir) + '.csv'
    df.to_csv(f_out, index=False)


if __name__ == '__main__':
    main()

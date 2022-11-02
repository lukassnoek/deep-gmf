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
@click.option('--binocular', is_flag=True, default=False)
def main(dataset_name, n_id_train, n_id_val, n_id_test, n_per_id, binocular):
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
        
    Notes
    -----
    In case of 128 IDs (2**7), the split (train/val/test) is 96/16/16;
    In case of 8192 IDs (2**13), the split (train/val/test) is 6144/1024/1024;
    In case of 16384 IDs (2**14), the split (train/val/test) is 12288/2048/2048;
    in case of 65536 IDs (2**16), the split (train/val/test) is 49152/8192/8192
    """

    data_dir = Path(f'/analyse/Project0257/lukas/data/{dataset_name}')

    if binocular:
        files = data_dir.glob('**/*_imageleft.jpg')
    else:
        files = data_dir.glob('**/*_image.jpg')
    info = defaultdict(list)  # keep track of features

    exp_total_files = sum([n_id_train, n_id_val, n_id_test]) * n_per_id    
    for i, f in tqdm(enumerate(files), total=exp_total_files):
        f = str(f)

        if binocular:
            feat = f.replace('_imageleft.jpg', '_features.h5')
            info['image_path_left'].append(f)
            f_right = f.replace('imageleft', 'imageright')
            if not Path(f_right).is_file():
                raise ValueError(f"File {f_right} does not exist!")
            info['image_path_right'].append(f_right)
        else:
            feat = f.replace('_image.jpg', '_features.h5')
            info['image_path'].append(f)

        if not Path(feat).is_file():
            raise ValueError(f"File {feat} does not exist!")

        info['feature_path'].append(feat)
        
        with h5py.File(feat, mode='r') as f_in:

            # Store all single value features as attributes
            for p in ('xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl',
                      'gender', 'ethn', 'age', 'id'):
                
                if p not in f_in.attrs.keys():
                    print(f"WARNING: attribute {p} does not exist in {feat}!")
                    to_add = 0
                else:
                    to_add = f_in.attrs[p]

                info[p].append(to_add)
                
        # Break if we want a subset of all IDs (for quick testing)
        if i > ((n_id_train + n_id_val + n_id_test) * n_per_id - 2):
            break

    df = pd.DataFrame.from_dict(info)
    df = df.sort_values(by=['id', 'gender', 'ethn', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'xl', 'yl', 'zl'], axis=0)

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
    df.to_csv(str(data_dir) + '.csv', index=False)
    
    
if __name__ == '__main__':
    main()

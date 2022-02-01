import sys
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

sys.path.append('.')
from src.models import MODELS
from src.io import create_dataset_test, DATASETS
from src.layers import CKA
from src.losses import AngleLoss

F_NAMES = {
    'bg': 'background',
    'l': 'lights',
    'xr': 'rot-x',
    'yr': 'rot-y',
    'zr': 'rot-z',
    'xt': 'trans-x',
    'yt': 'trans-y',
    'zt': 'trans-z',
    'xr_yr_zr': 'rotation',
    'xt_yt': 'translation',
    'ethn': 'ethnicity',
    'age': 'age',
    'id': 'Face ID'
}


@click.command()
@click.argument('model_name', type=click.Choice(MODELS.keys()))
@click.option('-t', '--target', type=click.STRING, default=['id'], multiple=True)
@click.option('-n', '--n-samples', type=click.INT, default=512)
@click.option('--n-id-test', type=click.INT, default=None)
@click.option('--cpu', is_flag=True)
def main(model_name, target, n_samples, n_id_test, cpu):

    info = pd.read_csv(Path(DATASETS['gmf']) / 'dataset_info.csv')
    info = info.sort_values(by=['bg', 'id', 'ethn', 'gender', 'age', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'l'],
                            axis=0)
    
    ctx = '/cpu:0' if cpu else '/gpu:0'
    with tf.device(ctx):
        
        dataset, df_test = create_dataset_test(info, n_samples=n_samples, batch_size=n_samples,
                                    n_id_test=n_id_test)

        X, tex, shape = dataset.__iter__().get_next()
        
        # Load model and filter layers
        full_model_name = f"{model_name}_dataset-gmf_target-{'+'.join(target)}"
        model = load_model(f'models/{full_model_name}', custom_objects={'AngleLoss': AngleLoss})
        
        layers = [l for l in model.layers if 'shortcut' not in l.name]
        factors = ['id', 'ethn', 'gender', 'age', 'bg', 'xr', 'yr', 'zr', 'xt', 'yt', 'zt', 'l']

        results = defaultdict(list)
        rdms = dict(feature=dict(), neural=dict())
        cka = CKA()
        for op_nr, layer in enumerate(tqdm(layers)):

            if 'layer' in layer.name:
                layer_nr = layer.name.split('_')[0].split('-')[1]
                op_type = layer.name.split('_')[-1]
            else:
                layer_nr = int(layer_nr) + 1
                op_type = 'logits'

            # Note to self: predict_step avoids warning that you get
            # when calling predict (but does the same)
            extractor = Model(inputs=model.inputs, outputs=[layer.output])
            a_N = extractor.predict_step(X)    
            a_N = tf.reshape(a_N, (n_samples, -1))            
            rdms['neural'][layer] = cka.get_rdm(a_N).numpy()

            for v in factors + [('shape', shape), ('tex', tex)]:
                if isinstance(v, str):
                    a_F = pd.get_dummies(df_test.loc[:, v]).to_numpy()
                else:
                    v, a_F = v

                a_F = tf.convert_to_tensor(a_F, dtype=tf.float32)
                r = np.round(cka(a_N, a_F).numpy(), 4)
                
                results['corr'].append(r)
                results['factor'].append(v)
                results['layer'].append(layer_nr)
                results['operation'].append(op_type)
                results['op_nr'].append(op_nr)

                if v not in rdms['feature'].keys():
                    rdms['feature'][layer] = cka.get_rdm(a_F).numpy()
                
        df = pd.DataFrame(results)
        df.to_csv(f'results/{full_model_name}.tsv', sep='\t', index=False)
        
        for tpe in ['feature', 'neural']:
            f_out = f'results/{full_model_name}_rdm-{tpe}.npz'
            this_dict = rdms[tpe]
            print(this_dict)
            np.savez(f_out, **this_dict)

if __name__ == '__main__':

    main()
import os
os.environ['DISPLAY'] = ':0.0'

import click
import random
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from GFG import Ctx
from GFG.model import Nf, Adata
from GFG.identity import IDModel
from GFG.core import Camera


lightopts = {
    'diffuse': 2,
    'specular': 3,
    'shadow': 4
}

ST = np.load('data/idm_St.npy')
SV = np.load('data/idm_Sv.npy')


@click.command('Main face generation API')
@click.option('--out-dir', default='gmf_', show_default=True, help='Output directory')
@click.option('--n-id', default=256, show_default=True, help='Number of face IDs to generate')
@click.option('--n-var', default=1, show_default=True, help='Number of variations per face ID')
@click.option('--add-background', is_flag=True, help='Add phase-scrambled background')
@click.option('--image-resolution', type=(int, int), default=(256, 256), show_default=True, help='Image resolution (pix.)')
@click.option('--image-format', type=click.Choice(['png', 'jpg']), default='png', show_default=True, help='Output format of image')
@click.option('--genders', type=click.Choice(['M', 'F', 'M+F']), default='M+F', show_default=True, help='Which gender to use')
@click.option('--ethns', type=click.Choice(['WC', 'EA', 'BA', 'WC+EA', 'WC+EA+BA']), default='WC+EA', show_default=True, help='Which ethnicity to use')
@click.option('--ages', type=(int, int), default=(25, 25), show_default=True, help='Which age range (min, max) to sample from')
@click.option('--shape-params', type=(float, float), default=(0., 1.), show_default=True, help='Parameters of normal dist (mu, std) to sample shape coefficients')
@click.option('--tex-params', type=(float, float), default=(0., 1.), show_default=True, help='Parameters of normal dist (mu, std) to sample texture coefficients')
@click.option('--x-rot', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation in X (deg.)')
@click.option('--y-rot', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation in Y (deg.)')
@click.option('--z-rot', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation in Z (deg.)')
@click.option('--x-trans', type=(int, int), default=(0, 0), show_default=True, help='Range to vary translation in X (mm.)')
@click.option('--y-trans', type=(int, int), default=(0, 0), show_default=True, help='Range to vary translation in Y (mm.)')
@click.option('--z-trans', type=(int, int), default=(0, 0), show_default=True, help='Range to vary translation in Z (mm.)')
@click.option('--x-rot-lights', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation of camera in X (deg.)')
@click.option('--y-rot-lights', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation of camera in Y (deg.)')
@click.option('--z-rot-lights', type=(int, int), default=(0, 0), show_default=True, help='Range to vary rotation of camera in Z (deg.)')
@click.option('--renderscale', default=4., show_default=True, help='Render scale of image')
@click.option('--camera-distance', default=400, show_default=True, help='Distance of camera from face')
@click.option('--light-source', default='./data/lights.yaml', show_default=True, help='Light source')
@click.option('--binocular', is_flag=True)
def main(out_dir, n_id, n_var, add_background, image_resolution, image_format, genders, ethns, ages,
         shape_params, tex_params, x_rot, y_rot, z_rot, x_trans, y_trans, z_trans, x_rot_lights,
         y_rot_lights, z_rot_lights, renderscale, camera_distance, light_source, binocular):

    ### Preliminary settings
    out_dir = Path(out_dir).absolute()
    if not out_dir.exists():
        raise ValueError(f"Output directory {str(out_dir)} does not exist!")

    IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
    idm = IDModel.load(IDM_PATH)
    tdet = np.load('/analyse/Project0294/GFG_data/tdet.npy')
    adata = Adata.load('quick_FACS_blendshapes_v2dense')
    base_nf = Nf.from_default()

    ### Set up context
    # Setup openGL context + camera
    ctx = Ctx(hidden=True)
    adata.attach(ctx)
    
    # Set custom lights with only a single source
    ctx.set_lights(Path(light_source))

    # Emo config
    emo_df = pd.read_csv('data/emo_config.tsv', sep='\t').set_index('emo')

    already_done = len(list(out_dir.glob('id-*')))
    for i_id in range(already_done, n_id):
        
        # ID params
        gend = random.choice(genders.split('+'))
        ethn = random.choice(ethns.split('+'))
        age = int(round(random.uniform(*ages)))
        shape_coeff = np.random.normal(*shape_params, size=len(idm)).astype(np.float16)
        tex_coeff = np.random.normal(*tex_params, size=(idm.nbands, len(idm))).astype(np.float16)
    
        nf = idm.generate(shape_coeff, tex_coeff, ethnicity=ethn, gender=gend, age=age,
                          basenf=base_nf, tdet=tdet)
        nf.attach(ctx)  # attach to openGL context

        id_name = str(i_id).zfill(len(str(n_id + 1)))
        this_out_dir = out_dir / f'id-{id_name}_gender-{gend}_ethn-{ethn}_age-{age}'
        if not this_out_dir.exists():
            this_out_dir.mkdir(parents=True)
        
        if n_var == 1:
            iter_ = range(n_var)
        else:
            iter_ = tqdm(range(n_var), desc=f'{i_id}/{n_id}')
        
        for i_var in iter_:
            
            f_out = str(this_out_dir / str(i_var).zfill(len(str(n_var))))
            
            xr = int(round(random.uniform(*x_rot)))
            yr = int(round(random.uniform(*y_rot)))
            zr = int(round(random.uniform(*z_rot)))
            xt = int(round(random.uniform(*x_trans)))
            yt = int(round(random.uniform(*y_trans)))
            zt = int(round(random.uniform(*z_trans)))
            xl = int(round(random.uniform(*x_rot_lights)))
            yl = int(round(random.uniform(*y_rot_lights)))
            zl = int(round(random.uniform(*z_rot_lights)))

            xt = xt * (1 - (zt / 1000))
            yt = yt * (1 - (zt / 1000))

            # Reset to default position and apply actual translation/rotation
            mu = nf.v[nf.groupvindex[base_nf.groupnames.index('face')]].mean(axis=0)
            mu[-1] = nf.v[nf.groupvindex[base_nf.groupnames.index('head')]].mean(axis=0)[-1]
            nf.skeleton.transform(x=xr, y=yr, z=zr, t=-mu, replace=True, order='txyz')
            nf.skeleton.transform(x=0, y=0, z=0, t=mu, replace=False)
            nf.skeleton.transform(x=0, y=0, z=0, t=[xt, yt, zt], replace=False)

            # Reset light position and rotate
            lights_pos = ctx.lights[0]._worldpos
            lights_pos[2] = camera_distance
            ctx.transform_lights(xl, yl, 0, lights_pos, replace=True)
            
            # Open background according to variation index (so it's counterbalanced across IDs)
            bg = np.array(Image.open(f'./data/background_{i_var+1}.png'))

            # Emotion
            emo = random.choice(emo_df.index)
            emo_amps = emo_df.loc[emo, :]
            for au, amp in emo_amps.items():
                adata.bshapes[au] = amp

            eyes = ['', 'left', 'right'] if binocular else ['']
            for eye in eyes:

                if eye == '':
                    offset = 0
                elif eye == 'left':
                    offset = -32
                elif eye == 'right':
                    offset = 32

                ctx._camera[0] = Camera(
                    ctx.win, image_resolution, renderscale,
                    target=[-11.5644, -13.0381, 0],
                    eye = [-11.5644 + offset, -13.0381, camera_distance],
                    up = [0, 1, 0],
                    FOV = 50,
                    near = 100.,
                    far = 1000.
                )
                ctx.assign_camera(0)
                # Render + alpha blend img & background
                img = ctx.render(dest='image')
                
                if add_background:
                    # Cast to float for normalization
                    img_arr = np.array(img).astype(np.float32)
                    img_rgb, img_a = img_arr[..., :3], img_arr[..., 3, None]

                    # for i in range(n_id):
                    #     bg = phase_scramble_image(
                    #         img_rgb.copy(), out_path=None, grayscale=False, shuffle_phase=False,
                    #         smooth=None, is_image=False
                    #     )
                    #     Image.fromarray(bg).save(f'data/background_{i+1}.png')

                    # exit()

                    img_a /= 255.  # alpha should be in 0-1 range!

                    # ... and alpha blend original image and background
                    img_arr = (img_rgb * img_a) + (bg * (1 - img_a))
                    img_arr = img_arr.astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    #bg = bg.astype(np.uint8)

                # Save to disk as an image; nice for inspection and may be
                # a little faster to load in Tensorflow compared to hdf5
                img.save(f_out + f'_image{eye}.{image_format}')

            # Save all (other) features as a hdf5 file
            with h5py.File(f_out + '_features.h5', 'w') as f_out_hdf:
                
                # save shape and texture parameters
                # Note that we're saving the "scaled" coefficients, i.e., the
                # coefficients x variance of the coeffients in PCA space
                f_out_hdf.create_dataset('shape', data=shape_coeff * SV, compression='gzip', compression_opts=9)
                f_out_hdf.create_dataset('tex', data=tex_coeff * ST, compression='gzip', compression_opts=9)            

                # Always save other generative parameters (rot, trans, lights, gender, ethn, age, id)
                for (name, p) in [
                    ('xr', xr), ('yr', yr), ('zr', zr), ('xt', xt), ('yt', yt), ('zt', zt + camera_distance),
                    ('xl', xl), ('yl', yl), ('zl', zl), ('emo', emo), ('gender', gend),
                    ('ethn', ethn), ('age', age), ('id', id_name)
                    ]:
                    f_out_hdf.attrs[name] = p

                if add_background:
                    f_out_hdf.attrs['bg'] = i_var

            for au, amp in emo_amps.items():
                adata.bshapes[au] = 0.

        nf.detach()


if __name__ == '__main__':
    main()

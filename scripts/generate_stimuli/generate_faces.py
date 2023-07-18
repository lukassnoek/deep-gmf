"""Utility CLI tool to generate batches of face stimuli from the GMF, with randomly
sampled ID parameters as well as random variations in pose, expression, lighting, etc."""

import os
os.environ['DISPLAY'] = ':0.0'  # necessary for headless rendering

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


ST = np.load('data/idm_St.npy')  # texture basis standard devs
SV = np.load('data/idm_Sv.npy')  # shape basis standard devs


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

    # Identity model (IDM) data
    IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
    idm = IDModel.load(IDM_PATH)
    tdet = np.load('/analyse/Project0294/GFG_data/tdet.npy')  # high freq texture
    adata = Adata.load('quick_FACS_blendshapes_v2dense')  # animation data (AUs)
    base_nf = Nf.from_default()  # nf = neutral face

    ### Set up context
    # Setup openGL context + camera
    ctx = Ctx(hidden=True)
    adata.attach(ctx)
    
    # Set custom lights with only a single source
    ctx.set_lights(Path(light_source))

    # Emo config
    emo_df = pd.read_csv('./data/emo_config.tsv', sep='\t').set_index('emo')

    # Check how many IDs have already been generated
    already_done = len(list(out_dir.glob('id-*')))

    # Loop from already_done to n_id
    for i_id in range(already_done, n_id):
        
        # Randomly sample identity parameters (gender, ethnicity, age)
        gend = random.choice(genders.split('+'))
        ethn = random.choice(ethns.split('+'))
        age = int(round(random.uniform(*ages))) 
        
        # Randomly sample shape and texture coefficients; note that these are by default
        # sampled from a standard normal distribution, which almost always results in
        # physically plausible faces; if you want more "extreme"/caraicatured faces,
        # sample from a wider distribution (e.g. N(0, 2))
        shape_coeff = np.random.normal(*shape_params, size=len(idm)).astype(np.float16)
        tex_coeff = np.random.normal(*tex_params, size=(idm.nbands, len(idm))).astype(np.float16)

        # Create a new neutral face (nf) with the sampled shape and texture coefficients
        # and identity parameters
        nf = idm.generate(shape_coeff, tex_coeff, ethnicity=ethn, gender=gend, age=age,
                          basenf=base_nf, tdet=tdet)
        nf.attach(ctx)  # attach to openGL context

        # Create output directory for this identity
        id_name = str(i_id).zfill(len(str(n_id + 1)))
        this_out_dir = out_dir / f'id-{id_name}_gender-{gend}_ethn-{ethn}_age-{age}'
        if not this_out_dir.exists():
            this_out_dir.mkdir(parents=True)
        
        # For each identity, loop over n_var and randomly sample variation parameters
        iter_ = tqdm(range(n_var), desc=f'{i_id}/{n_id}')        
        for i_var in iter_:
            
            f_out = str(this_out_dir / str(i_var).zfill(len(str(n_var))))
            
            xr = int(round(random.uniform(*x_rot)))  # rotation in X
            yr = int(round(random.uniform(*y_rot)))  # rotation in Y
            zr = int(round(random.uniform(*z_rot)))  # rotation in Z
            xt = int(round(random.uniform(*x_trans)))  # translation in X
            yt = int(round(random.uniform(*y_trans)))  # translation in Y
            zt = int(round(random.uniform(*z_trans)))  # translation in Z
            xl = int(round(random.uniform(*x_rot_lights)))  # rotation of light in X
            yl = int(round(random.uniform(*y_rot_lights)))  # rotation of light in Y
            zl = int(round(random.uniform(*z_rot_lights)))  # rotation of light in Z

            # Note: probably best to sample only X and Y in light direction

            # Little hack to scale the translation in the X and Y direction with the
            # distance of the camera to the face (determined by zt), such that the face
            # stays within the image
            xt = xt * (1 - (zt / 1000))
            yt = yt * (1 - (zt / 1000))

            # We want to rotate around the center of the face, not the center of the
            # coordinate system, so we need to compute the center of the face (mu) by 
            # computing the mean of the face vertices
            mu = nf.v[nf.groupvindex[base_nf.groupnames.index('face')]].mean(axis=0)
            
            # However, the mean of Z (depth) of the face is not really the center of the
            # head, so we compute this separately by taking the mean of the Z of the head
            # vertices
            mu[-1] = nf.v[nf.groupvindex[base_nf.groupnames.index('head')]].mean(axis=0)[-1]
            
            # Now, translate face to origin and rotate
            nf.skeleton.transform(x=xr, y=yr, z=zr, t=-mu, replace=True, order='txyz')
            
            # Now, translate face back to original position
            nf.skeleton.transform(x=0, y=0, z=0, t=mu, replace=False)
            
            # And apply whatever translation parameters were sampled
            nf.skeleton.transform(x=0, y=0, z=0, t=[xt, yt, zt], replace=False)

            # Reset light position and rotate
            lights_pos = ctx.lights[0]._worldpos
            lights_pos[2] = camera_distance
            ctx.transform_lights(xl, yl, 0, lights_pos, replace=True)
            
            # Open background according to variation index (so it's counterbalanced across IDs)
            bg = np.array(Image.open(f'./data/background_{i_var+1}.png'))

            # Set AUs with sampled emotion
            emo = random.choice(emo_df.index)
            emo_amps = emo_df.loc[emo, :]
            for au, amp in emo_amps.items():
                adata.bshapes[au] = amp

            # If we want "binocular" stimuli, we render the face three times: once with
            # a camera in the center (pointed directly at the center of the coordinate space),
            # one simulating a left eye (offset 32 mm to the left), and one simulating a
            # right eye (offset 32 mm to the right). If we don't want binocular stimuli,
            # we only render once with the camera in the center.
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

                # Finally, actually render the face image
                img = ctx.render(dest='image')
                
                if add_background:
                    
                    # Cast to float for normalization
                    img_arr = np.array(img).astype(np.float32)
                    
                    # Split into RGB and alpha channel
                    img_rgb, img_a = img_arr[..., :3], img_arr[..., 3, None]

                    img_a /= 255.  # alpha should be in 0-1 range!

                    # ... and alpha blend original image and background
                    img_arr = (img_rgb * img_a) + (bg * (1 - img_a))
                    img_arr = img_arr.astype(np.uint8)
                    img = Image.fromarray(img_arr)

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

            # Reset blendshapes to neutral expression (i.e., all AUs have amplitude 0)
            for au, amp in emo_amps.items():
                adata.bshapes[au] = 0.

        # Detach neutral face from context
        nf.detach()


if __name__ == '__main__':
    main()

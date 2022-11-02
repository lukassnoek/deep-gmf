import os
os.environ['DISPLAY'] = ':0.0'

import click
import random
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import GFG
from GFG import Ctx
from GFG.model import Nf
from GFG.identity import IDModel
from GFG.core import Camera
from ogbGL.utils import imresize
from generate_backgrounds import phase_scramble_image


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
@click.option('--save-image-separately', is_flag=True, help='Save image as file (not as part of hdf5 file)')
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
@click.option('--save-id-params', is_flag=True, help='Save ID parameters (gender, ethn, age, shape, tex)')
@click.option('--save-background', is_flag=True, help='Save background (in hdf5 file)')
@click.option('--save-buffers', is_flag=True, help='Save intermediate buffers (in hdf5 file)')
@click.option('--save-lighting', is_flag=True, help='Save intermediate lighting (in hdf5 file)')
@click.option('--renderscale', default=4., show_default=True, help='Render scale of image')
@click.option('--camera-distance', default=400, show_default=True, help='Distance of camera from face')
@click.option('--light-source', default='data/lights.yaml', show_default=True, help='Light source')
def main(out_dir, n_id, n_var, add_background, image_resolution, image_format, save_image_separately, genders, ethns, ages,
         shape_params, tex_params, x_rot, y_rot, z_rot, x_trans, y_trans, z_trans, x_rot_lights,
         y_rot_lights, z_rot_lights, save_id_params, save_background, save_buffers, save_lighting, renderscale,
         camera_distance, light_source):

    ### Preliminary settings
    out_dir = Path(out_dir).absolute()
    if not out_dir.exists():
        raise ValueError(f"Output directory {str(out_dir)} does not exist!")

    IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
    idm = IDModel.load(IDM_PATH)
    tdet = np.load('/analyse/Project0294/GFG_data/tdet.npy')

    base_nf = Nf.from_default()

    ### Set up context
    # Setup openGL context + camera
    ctx = Ctx(hidden=True)

    # Set custom lights with only a single source
    ctx.set_lights(Path(light_source))

    for i_id in tqdm(range(n_id), desc='ID gen'):

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
            iter_ = tqdm(range(n_var), desc='Var gen')
        
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
            
            for eye in ['left', 'right']:
                offset = 32 if eye == 'right' else -32

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
                    img_a /= 255.  # alpha should be in 0-1 range!

                    # Generate background ...
                    bg = phase_scramble_image(
                        img_rgb.copy(), out_path=None, grayscale=False, shuffle_phase=False,
                        smooth=None, is_image=False
                    )
                    # ... and alpha blend original image and background
                    img_arr = (img_rgb * img_a) + (bg * (1 - img_a))
                    img_arr = img_arr.astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    bg = bg.astype(np.uint8)

                if save_image_separately:
                    # Save to disk as an image; nice for inspection and may be
                    # a little faster to load in Tensorflow compared to hdf5
                    img.save(f_out + f'_image{eye}.{image_format}')

                # Save all (other) features as a hdf5 file
                with h5py.File(f_out + '_features.h5', 'w') as f_out_hdf:
                    
                    # save shape and texture parameters
                    if save_id_params:  
                        # Note that we're saving the "scaled" coefficients, i.e., the
                        # coefficients x variance of the coeffients in PCA space
                        f_out_hdf.create_dataset('shape', data=shape_coeff * SV, compression='gzip', compression_opts=9)
                        f_out_hdf.create_dataset('tex', data=tex_coeff * ST, compression='gzip', compression_opts=9)            

                    if not save_image_separately:  # save img, too, if not already
                        f_out_hdf.create_dataset('img', data=np.array(img), compression='gzip', compression_opts=9)

                    if save_background:  # save custom background
                        f_out.create_dataset('background', data=bg, compression='gzip', compression_opts=9)

                    if save_buffers:
                        buffers = {
                            'frag_pos': [ctx.fbo['gBuffer'], 'gPosition', [0,1,2]],  # fragment shader position
                            'world_pos': [ctx.fbo['gBuffer'], 'gWPosition', [0,1,2]],  # world position
                            'spec_normal': [ctx.fbo['gBuffer'], 'gNormal', [0,1,2]],  # specular normals
                            'diff_normal': [ctx.fbo['gBuffer'], 'gBlurNormal', [0,1,2]],  # diffuse normals
                            'albedo': [ctx.fbo['gBuffer'], 'gAlbedoSpec', [0,1,2]],  # texture
                            'ssao': [ctx.fbo['SSAOblur'], 'ctbuffer', [0]]  # ambient occlusion
                        }

                        for name, bufflist in buffers.items():
                            ctx._image_pass(bufflist[0], bufflist[1])
                            if isinstance(ctx.im, list):
                                im_ = np.dstack([np.array(im) for i, im in enumerate(ctx.im)
                                                                if i in bufflist[2]])
                            else:
                                im_ = np.array(ctx.im)
                                
                            f_out_hdf.create_dataset(name, data=im_, compression='gzip', compression_opts=9)                        

                    if save_lighting:
                        for name, opt in lightopts.items():
                            ctx.programs['LightingShader']['out_type'] = opt
                            ctx._lighting_pass()
                            im_ = np.array(ctx.dispatch_draw('image'))
                            if im_.shape[0] == (image_resolution[0] * renderscale):
                                # Some lighting data is in upsampled resolution; downsample first!
                                im_ = imresize(im_, image_resolution, resample=Image.BILINEAR)

                            f_out_hdf.create_dataset(name, data=im_, compression='gzip', compression_opts=9)
                        
                        ctx.fbo['draw'].bind()
                        GFG.GL.glClearColor(0.0,0.0,0.0,0.0)
                        GFG.GL.glClear(GFG.GL.GL_COLOR_BUFFER_BIT)

                        # now run fowrard pass and return image
                        ctx._forward_pass()
                        im_ = np.array(ctx.dispatch_draw('image'))
                        f_out_hdf.create_dataset('glass', data=im_, compression='gzip', compression_opts=9)

                        ctx.programs['LightingShader']['out_type'] = 1
                        ctx.programs['LightingShader'].update_uniforms()                    

                    # Always save other generative parameters (rot, trans, lights, gender, ethn, age, id)
                    for (name, p) in [
                        ('xr', xr), ('yr', yr), ('zr', zr), ('xt', xt), ('yt', yt), ('zt', zt),
                        ('xl', xl), ('yl', yl), ('zl', zl), ('gender', gend), ('ethn', ethn), ('age', age),
                        ('id', id_name)
                        ]:
                        f_out_hdf.attrs[name] = p

        nf.detach()


if __name__ == '__main__':
    main()

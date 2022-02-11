import os
os.environ['DISPLAY'] = ':0.0'

import h5py
import shutil
import random
import numpy as np
import pandas as pd
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

### SETUP
ROOT = Path(__file__).parents[2].absolute()  # ~/deep-gmf
OUT_DIR = Path('/analyse/Project0257/lukas/data/gmf_manyIDsingleIMGnovariation2')

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load identity model + base nf
IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(IDM_PATH)
# For extra detail; used in `idm.generate`
tdet = np.load(OUT_DIR.parent / 'tdet.npy')

#adata = Adata.load('quick_FACS_blendshapes_v2dense')
base_nf = Nf.from_default()

# Setup openGL context + camera
ctx = Ctx(hidden=True)

# To init a camera, a model should exist, so using base_nf
base_nf.attach(ctx)
ctx._camera[0] = Camera(
    ctx.win, (256, 256), 4.,  # res, renderscale
    target=[-11.5644, -13.0381, 0],
    # Increase distance to 700 (fits face in FOV)
    eye = [-11.5644, -13.0381, 400],
    up = [0, 1, 0],
    FOV = 50,
    near = 100.,
    far = 1000.
)
ctx.assign_camera(0)
base_nf.detach()  # not necessary anymore

### PARAMETERS

# ID params
GENDERS = ('M', 'F')
ETHNS = ('WC', 'EA')
AGES = (20, 50)  # min, max

# Rotations (RS) and translations (TS)
XRS = (-30, 30)  # min, max
YRS = (-30, 30)
ZRS = (-30, 30)
XTS = (-100, 100)
YTS = (-100, 100)
ZTS = (-450, 50)

# Light rotations
XYZ_LIGHTS = [
    (-45, 45),  # x (above -> in front -> below)
    (-45, 45)   # y (left  -> in front -> right)
]
ctx.set_lights(Path(ROOT / 'lights.yaml'))
        
# lighting shader output options:
# {type: lighting shader option}
lightopts = {
#    'diffuse': 2,
#    'specular': 3,
#    'shadow': 4
}

###  MAIN GENERATION LOOP

# Loop across ID parameters
current_id = 0
for i in tqdm(range(524288)):  # 2**16
    gender = random.choice(GENDERS)
    ethn = random.choice(ETHNS)
    age = int(round(random.uniform(*AGES)))
    
    f_out = str(OUT_DIR / f'id-{str(current_id).zfill(7)}_gender-{gender}_ethn-{ethn}_age-{age}')

    # Generate vertex and texture parameters
    v_coeff = np.random.normal(0, 1, size=len(idm)).astype(np.float16)
    t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm))).astype(np.float16)
    #np.savez(str(this_out_dir) + '.npz', v_coeff=v_coeff, t_coeff=t_coeff)
        
    # Generate neutral face with ID parameters
    nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age,
                      basenf=base_nf, tdet=tdet)
    nf.attach(ctx)  # attach to openGL context

    xr = 0#round(random.uniform(*XRS), 2)
    yr = 0#round(random.uniform(*YRS), 2)
    zr = 0#round(random.uniform(*ZRS), 2)
    xt = 0#round(random.uniform(*XTS), 2)
    yt = 0#round(random.uniform(*YTS), 2)
    zt = 0#round(random.uniform(*ZTS), 2)
    lx = 0#round(random.uniform(*XYZ_LIGHTS[0]), 2)
    ly = 0#round(random.uniform(*XYZ_LIGHTS[1]), 2)
            
    # Reset to default position and apply actual translation/rotation
    nf.transform_model(0, 0, 0, [0, 0, 0], order='txyz', replace=True)
    nf.transform_model(xr, yr, zr, [xt, yt, zt], order='xyzt', replace=False)

    # Set custom lights with only a single source
    ctx.transform_lights(lx, ly, 0, [0, 0, 0], replace=True)

    # Specify buffers to extract
    # {type: [framebuffer, attachment, channels]}
    buffers = {
        'frag_pos': [ctx.fbo['gBuffer'], 'gPosition', [0,1,2]],
        'world_pos': [ctx.fbo['gBuffer'], 'gWPosition', [0,1,2]],
        'spec_normal': [ctx.fbo['gBuffer'], 'gNormal', [0,1,2]],
        'diff_normal': [ctx.fbo['gBuffer'], 'gBlurNormal', [0,1,2]],
        'albedo': [ctx.fbo['gBuffer'], 'gAlbedoSpec', [0,1,2]],
        'ssao': [ctx.fbo['SSAOblur'], 'ctbuffer', [0]]
    }  # note: should be done 

    # Render + alpha blend img & background
    img_orig = ctx.render(dest='image')
    # Cast to float for normalization
    img_arr = np.array(img_orig).astype(np.float32)
    img_rgb, img_a = img_arr[..., :3], img_arr[..., 3, None]
    img_a /= 255.  # alpha should be in 0-1 range!

    # Generate background ...
    bg_rgb = phase_scramble_image(img_rgb, out_path=None, grayscale=False, shuffle_phase=False,
                                    smooth=None, is_image=False)
    # ... and alpha blend original image and background
    img_arr = (img_rgb * img_a) + (bg_rgb * (1 - img_a))

    # Save to disk
    img_arr = img_arr.astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save(f_out + '_image.png')
    
    bg_rgb = bg_rgb.astype(np.uint8)
    other_data = {}#'background': bg_rgb}
    for name, bufflist in buffers.items():
        ctx._image_pass(bufflist[0], bufflist[1])
        if isinstance(ctx.im, list):
            other_data[name] = np.dstack([np.array(im) for i, im in enumerate(ctx.im)
                                            if i in bufflist[2]])
        else:
            other_data[name] = np.array(ctx.im)
    
    # step 3 - extract the lighting layers
    for name, opt in lightopts.items():
        ctx.programs['LightingShader']['out_type'] = opt
        ctx._lighting_pass()
        im = ctx.dispatch_draw('image')
        other_data[name] = np.array(im)
    
    # step 4 - extract the glass layer
    # we need to first clear the draw buffer
    ctx.fbo['draw'].bind()
    GFG.GL.glClearColor(0.0,0.0,0.0,0.0)
    GFG.GL.glClear(GFG.GL.GL_COLOR_BUFFER_BIT)

    # now run fowrard pass and return image
    ctx._forward_pass()
    im = ctx.dispatch_draw('image')
    other_data['glass'] = np.array(im)

    with h5py.File(f_out + '_features.h5', 'w') as f_out:
        # for prop, dat in other_data.items():
        #     if dat.shape[:2] == (1024, 1024):
        #         # Buffers need to resized when renderscale > 1
        #         dat = imresize(dat, (256, 256), resample=Image.BILINEAR)
        #     f_out.create_dataset(prop, data=dat.squeeze(), compression='lzf')
    
        f_out.create_dataset('shape', data=v_coeff, compression='gzip', compression_opts=9)
        f_out.create_dataset('tex', data=t_coeff, compression='gzip', compression_opts=9)            

        for (name, p) in [
            ('xr', xr), ('yr', yr), ('zr', zr), ('xt', xt), ('yt', yt), ('zt', zt),
            ('lx', lx), ('ly', ly), ('gender', gender), ('ethn', ethn), ('age', age),
            ('id', str(current_id).zfill(7))
            ]:
            f_out.attrs[name] = p

    ctx.programs['LightingShader']['out_type'] = 1
    ctx.programs['LightingShader'].update_uniforms()

    nf.detach()
    current_id += 1
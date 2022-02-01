import os
# necessary when on deepnet server
os.environ['DISPLAY'] = ':0.0'

import GFG
import h5py
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from pathlib import Path
from GFG import Ctx
from GFG.model import Nf
from GFG.identity import IDModel
from GFG.core import Camera
from generate_backgrounds import phase_scramble_image


ROOT = Path(__file__).parents[2].absolute()
OUT_DIR = Path('/analyse/Project0257/lukas/data/gmf_random')

# Load identity model + base nf
IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(IDM_PATH)
#adata = Adata.load('quick_FACS_blendshapes_v2dense')
base_nf = Nf.from_default()

### SETUP CAMERA + DEFINITION PARAMETERS
# Setup openGL context + camera
ctx = Ctx(hidden=True)

# lighting shader output options:
# {type: lighting shader option}
lightopts = {
    'diffuse': 2,
    'specular': 3,
    'shadow': 4
}

# Rotations (RS) and translations (TS)
XRS = (-45, 45)  # min, max
YRS = (-45, 45)
ZRS = (-45, 45)
XTS = (-100, 100)
YTS = (-100, 100)
ZTS = (-220, 100)

# Light rotations
XYZ_LIGHTS = [
    (-90, 90),  # x (above -> in front -> below)
    (-90, 90)   # y (left  -> in front -> right)
]

# For extra detail; used in `idm.generate`
tdet = np.load(OUT_DIR.parent / 'tdet.npy')

#%%  MAIN GENERATION LOOP
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Loop across ID parameters
current_id = 0
for i in range(2):
    gender = np.random.choice(['M', 'F'])
    ethn = np.random.choice(['WC', 'EA', 'BA'])
    age = int(round(np.random.uniform(20, 60)))
    
    this_out_dir = OUT_DIR / f'id-{str(current_id).zfill(3)}_gender-{gender}_ethn-{ethn}_age-{age}'
        
    # Generate vertex and texture parameters
    v_coeff = np.random.normal(0, 1, size=len(idm))
    t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))
    np.savez(str(this_out_dir) + '.npz', v_coeff=v_coeff, t_coeff=t_coeff)
        
    # Generate neutral face with ID parameters
    nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age,
                      basenf=base_nf, tdet=tdet)
    nf.attach(ctx)  # attach to openGL context
    ctx._camera[0] = Camera(
        ctx.win, (256, 256), 4.,  # res, renderscale
        target=[-11.5644, -13.0381, 0],
        # Increase distance to 700 (fits face in FOV)
        eye = [-11.5644, -13.0381, 700],
        up = [0, 1, 0],
        FOV = 50,
        near = 100.,
        far = 1000.
    )
    ctx.assign_camera(0)
    ctx.transform_lights(0, 0, 0, order='xyz')  # Reset lights

    params = defaultdict(list)
    for ii in range(10):
        xr = round(np.random.uniform(*XRS), 2)
        yr = round(np.random.uniform(*YRS), 2)
        zr = round(np.random.uniform(*ZRS), 2)
        xt = round(np.random.uniform(*XTS), 2)
        yt = round(np.random.uniform(*YTS), 2)
        zt = round(np.random.uniform(*ZTS), 2)
        lx = round(np.random.uniform(*XYZ_LIGHTS[0]), 2)
        ly = round(np.random.uniform(*XYZ_LIGHTS[1]), 2)
        
        for (name, p) in [('xr', xr), ('yr', yr), ('zr', zr), ('xt', xt), ('yt', yt), ('zt', zt),
                          ('lx', lx), ('ly', ly), ('gender', gender), ('ethn', ethn), ('age', age),
                          ('id', str(current_id).zfill(3))]:
            params[name].append(p)
        
        # Reset to default position and apply actual translation/rotation
        nf.transform_model(0, 0, 0, [0, 0, 0], order='txyz', replace=True)
        nf.transform_model(xr, yr, zr, [xt, yt, zt], order='xyzt', replace=False)

        # Set custom lights with only a single source
        ctx.set_lights(Path(ROOT / 'lights.yaml'))
        ctx.transform_lights(lx, ly, 0, [0, 0, 0])  # reset!

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
        print(img_arr.shape)
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
        f_out = this_out_dir / f'{str(ii).zfill(5)}_image.png'
        Path(f_out).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(f_out))  # change!!!
        
        other_data = {'background': bg_rgb.astype(np.uint)}
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
        
        for name, dat in other_data.items():
            if dat.max() <= 1:
                dat *= 255
            
            dat = dat.squeeze().astype(np.uint8)
            
            this_img = Image.fromarray(dat)
            this_img.save(str(f_out).replace('_image.png', f'_{name}.png'))
        
        #with h5py.File(str(f_out).replace('_image.png', '_features.h5'), 'w') as f_out:
        #    for prop, dat in other_data.items():
        #        f_out.create_dataset(prop, data=dat)
        
        ctx.programs['LightingShader']['out_type'] = 1
        ctx.programs['LightingShader'].update_uniforms()

    params = pd.DataFrame(params)
    params.to_csv(str(this_out_dir) + '.tsv', index=False)    
    nf.detach()
    current_id += 1
#%% IMPORTS + DATA LOADING
import os
os.environ['DISPLAY'] = ':0.0'
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from itertools import product
from tqdm import tqdm
from skimage import transform
from GFG import Ctx
from GFG.model import Nf, Adata
from GFG.identity import IDModel
from ogbGL.draw_objects import Camera


ROOT = Path(__file__).parents[1]
OUT_DIR = Path('/analyse/Project0257/lukas/data/gmf')
#OUT_DIR = ROOT / 'data' / 'gmf2'

# Load identity model + base nf
IDM_PATH = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(IDM_PATH)
adata = Adata.load('quick_FACS_blendshapes_v2dense')
base_nf = Nf.from_default()

#%%  SETUP CAMERA + DEFINITION PARAMETERS

# Setup openGL context + camera
ctx = Ctx(res=(256, 256), renderscale=4.)
ctx.camera[0] = Camera(
    ctx.win, ctx._ar,
    target = [-11.5644, -13.0381, 0],
    # Increase distance to 700 (fits face in FOV)
    eye = [-11.5644, -13.0381, 700],
    up = [0,1,0],
    FOV = 50,
    near = 100.0,
    far = 1000.0
)
# Reset lights
ctx.transform_lights(0, 0, 0, order='xyz')
adata.attach(ctx)

# Generation parameters
N_IDS = 8
GENDERS = ['F', 'M']
ETHNS = ['WC', 'BA', 'EA']
AGES = [20, 40]
BGS = [0, 1]  # backgrounds

# Rotations (RS) and translations (TS)
XRS = [(0, -45), (1, 0), (2, 45)]
YRS = [(0, -45), (1, 0), (2, 45)]
ZRS = [(0, -45), (1, 0), (2, 45)]
XTS = [(0, -55), (1, 0), (2, 55)]
YTS = [(0, -55), (1, 0), (2, 55)]
ZTS = [(0, -220), (1, -60), (2, 100)]

# Expressions
#EXPS = [(0, ('AU6', 'AU12')), (1, []), (2, ('AU9', 'AU10Open'))]

# Light rotations
XYZ_LIGHTS = [
    (0, (0, 0, 0)),    # in front
    (1, (-90, 0, 0)),  # above 
    (2, (+90, 0, 0)),  # below
    (3, (0, -90, 0)),  # left
    (4, (0, +90, 0))   # right
]

# For extra detail; used in `idm.generate`
tdet = np.load(OUT_DIR.parent / 'tdet.npy')
#tdet = np.load('/analyse/Project0257/lukas/data/tdet.npy')

# Pre-load background images, to be blended in image later
bg_imgs = [np.array(Image.open(ROOT / 'data' / f'background_{i}.png'))
           for i in range(1, 4)]

#%%  MAIN GENERATION LOOP
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Loop across ID parameters
id_params = product(GENDERS, ETHNS, AGES)
current_id = 0
for gender, ethn, age in id_params:
    
    # Create `N_IDS` identities with ID parameters
    for _ in tqdm(range(N_IDS)):

        this_out_dir = OUT_DIR / f'id-{str(current_id).zfill(3)}_gender-{gender}_ethn-{ethn}_age-{age}'
        
        # Generate vertex and texture parameters
        v_coeff = np.random.normal(0, 1, size=len(idm))
        t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))
        np.savez(str(this_out_dir) + '.npz', v_coeff=v_coeff, t_coeff=t_coeff)
        
        # Generate neutral face with ID parameters
        nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age,
                          basenf=base_nf, tdet=tdet)
        nf.attach(ctx)  # attach to openGL context
        
        stim_params = product(BGS, XRS, YRS, ZRS, ZTS, XYZ_LIGHTS)#, EXPS)
        for bg, xr, yr, zr, zt, (i_xyzl, xyzl) in stim_params:#, (i_exp, exp) in stim_params:
            (ixr, xr), (iyr, yr), (izr, zr) = [xr, yr, zr]
            (izt, zt) = zt
        
            #for au in exp:
            #    adata.bshapes[au] = 1.
        
            # Reset to default position and apply actual translation/rotation
            nf.transform_model(0, 0, 0, [0, 0, 0], order='txyz', replace=True)
            nf.transform_model(xr, yr, zr, [0, 0, zt], order='xyzt', replace=False)

            ctx.set_lights(Path(ROOT / 'lights.yaml'))
            ctx.transform_lights(*xyzl, [0, 0, 0])

            # Render + alpha blend img & background
            img_orig = ctx.render(dest='image')

            for (ixt, xt), (iyt, yt) in product(XTS, YTS):
                img_arr = np.array(img_orig)
                tform = transform.EuclideanTransform(translation=(xt, yt))
                img_arr = transform.warp(img_arr, tform.inverse)
                img_arr *= 255

                img_rgb, img_a = img_arr[..., :3], img_arr[..., 3, None] / 255.
                bg_rgb = bg_imgs[bg]
                img_arr = (img_rgb * img_a) + (bg_rgb * (1 - img_a))  # alpha blend
                img_arr = np.dstack((img_arr, np.ones((256, 256)) * 255)).astype(np.uint8)
            
                # Save to disk
                img = Image.fromarray(img_arr)
                f_out = this_out_dir / f'bg-{bg}_xr-{ixr}_yr-{iyr}_zr-{izr}_xt-{ixt}_yt-{iyt}_zt-{izt}_l-{i_xyzl}.jpg'#_exp-{i_exp}.png'
                Path(f_out).parent.mkdir(parents=True, exist_ok=True)
                img.convert('RGB').save(str(f_out))  # convert RGBA -> RGB in order to save as JPG
            
            #for au in exp:
            #    adata.bshapes[au] = 0.
            
        nf.detach()
        current_id += 1
# %%

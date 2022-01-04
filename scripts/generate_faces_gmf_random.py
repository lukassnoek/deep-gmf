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
from generate_backgrounds import phase_scramble_image

ROOT = Path(__file__).parents[1]
OUT_DIR = Path('/analyse/Project0257/lukas/data/gmf_random')
#OUT_DIR = ROOT / 'data' / 'gmf_random'

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

# Rotations (RS) and translations (TS)
# Note: no translations in Z for now
XRS = (-45, 45)
YRS = (-45, 45)
ZRS = (-45, 45)
XTS = (-55, 55)
YTS = (-55, 55)
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
for i in range(1000):
    gender = np.random.choice(['M', 'F'])
    ethn = np.random.choice(['WC', 'EA', 'BA'])  
    age = np.random.uniform(20, 60)
    
    this_out_dir = OUT_DIR / f'id-{str(current_id).zfill(3)}_gender-{gender}_ethn-{ethn}_age-{age}'
        
    # Generate vertex and texture parameters
    v_coeff = np.random.normal(0, 1, size=len(idm))
    t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))
    np.savez(str(this_out_dir) + '.npz', v_coeff=v_coeff, t_coeff=t_coeff)
        
    # Generate neutral face with ID parameters
    nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age,
                      basenf=base_nf, tdet=tdet)
    nf.attach(ctx)  # attach to openGL context
        
    for ii in range(8192):  # 2**13
        xr = np.random.uniform(*XRS)
        yr = np.random.uniform(*YRS)
        zr = np.random.uniform(*ZRS)
        xt = np.random.uniform(*XTS)
        yt = np.random.uniform(*YTS)
        zt = np.random.uniform(*ZTS)
        lx = np.random.uniform(*XYZ_LIGHTS[0])
        ly = np.random.uniform(*XYZ_LIGHTS[1])
        
        # Reset to default position and apply actual translation/rotation
        nf.transform_model(0, 0, 0, [0, 0, 0], order='txyz', replace=True)
        nf.transform_model(xr, yr, zr, [0, 0, zt], order='xyzt', replace=False)

        ctx.set_lights(Path(ROOT / 'lights.yaml'))
        #ctx.transform_lights(lx, ly, 0, [0, 0, 0])

        # Render + alpha blend img & background
        img_orig = ctx.render(dest='image')
        img_arr = np.array(img_orig)
        tform = transform.EuclideanTransform(translation=(xt, yt))
        img_arr = transform.warp(img_arr, tform.inverse)
        img_arr *= 255

        img_rgb, img_a = img_arr[..., :3], img_arr[..., 3, None] / 255.
        bg_rgb = phase_scramble_image(img_rgb, out_path=None, grayscale=False, shuffle_phase=False,
                                      smooth=None, is_image=False)
        img_arr = (img_rgb * img_a) + (bg_rgb * (1 - img_a))  # alpha blend

        # Save to disk
        img_arr = img_rgb.astype(np.uint8)
        img = Image.fromarray(img_arr)
        f_out = this_out_dir / f'{str(ii).zfill(5)}.jpg'
        Path(f_out).parent.mkdir(parents=True, exist_ok=True)
        #img.convert('RGB').save(str(f_out))
        img.save(str(f_out)) 
                      
    nf.detach()
    current_id += 1
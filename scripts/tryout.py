#%%
import os
os.environ['DISPLAY'] = ':0.0'
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from GFG import Ctx
from GFG.model import Nf
from GFG.identity import IDModel
from ogbGL.draw_objects import Camera

np.random.seed(42)
try:
    ROOT = Path(__file__).parents[1]
except:
    ROOT = Path('/home/lukass/deep-gmf')

def _create_filename(i, gender, ethn, age, xr, yr, zr, xt, yt):
    f_out = str(ROOT / f'data/gender-{gender}/ethn-{ethn}/age-{age}/id-{str(i+1).zfill(4)}')
    
    for n, p in {'xr': xr, 'yr': yr, 'zr': zr, 'xt': xt, 'yt': yt, 'zt': zt}.items():
        if isinstance(p, (int, float)):
            if p == 0:
                p = f'n{p}'
            elif p > 0:
                p = f'p{p}'
            else:
                p = f'm{abs(p)}'
                        
        f_out += f'_{n}-{p}'
    
    f_out += '.png'
    return f_out            

# Setup openGL context + camera
ctx = Ctx(res=(256, 256), renderscale=4.)
ctx.camera[0] = Camera(
    ctx.win, ctx._ar,
    target = [-11.5644, -13.0381, 0],
    eye = [-11.5644, -13.0381, 450.],
    up = [0,1,0],
    FOV = 50,
    near = 100.0,
    far = 1000.0
)
ctx.transform_lights(0, 0, 0, order='xyz')

# Load identity model
data_path = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(data_path)
base_nf = Nf.from_default()

#%%
# Generation parameters
N_IDS = 1
x_rots = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
y_rots = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
z_rots = [0, 45, 90, 135, 180, 225, 270, 315]
x_trans = [-50, -25, 0, 25, 50]
y_trans = [-50, -25, 0, 25, 50]
z_trans = [0]#-300, -200, -100, 0, 100]

if (ROOT / 'data' / 'gender-M').exists():
    shutil.rmtree(ROOT / 'data' / 'gender-M')

gender, ethn, age = 'M', 'WC', 25
    
v_coeff = np.random.normal(0, 1, size=len(idm))
t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))

tdet = np.load(ROOT / 'GFG_data' / 'tdet.npy')
nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age,
                  basenf=base_nf, tdet=tdet)
nf.attach(ctx)
head_idx = nf.groupvindex[nf.groupnames.index('head')] - 1
com = nf.v[head_idx, :].mean(axis=0)  # center of mass

out_dir = ROOT / f'data/gender-{gender}/ethn-{ethn}/age-{age}/id-{str(1).zfill(4)}'
for ixr, xr in enumerate(x_rots):
    for iyr, yr in enumerate(y_rots):
        for izr, zr in enumerate(z_rots):
            for ixt, xt in enumerate(x_trans):
                for iyt, yt in enumerate(y_trans):
                    for izt, zt in enumerate(z_trans):
                        f_out = out_dir / f'xr-{ixr}_yr-{iyr}_zr-{izr}_xt-{ixt}_yt-{iyt}_zt-{izt}.png'
                        # Translate to origin, rotate, and apply actual translation
                        nf.transform_model(0, 0, 0, [0, 0, 0], replace=True)
                        nf.transform_model(xr, yr, zr, -com, order='txyz', replace=True)
                        nf.transform_model(0, 0, 0, np.array([xt, yt, zt]), order='t', replace=False)

                        # Render + save
                        img = ctx.render(dest='image')
                        Path(f_out).parent.mkdir(parents=True, exist_ok=True)
                        img.save(str(f_out))

nf.detach()
                    
                    
# tf.transform_model(x, y, z, tx, ty, tz, scale, order='xyzts')
#ctx.set_lights()
# from ogbGL.draw_objects import Light
# new_light = Light(ctx.win)
# ctx.lights.append(new_light)
# %%

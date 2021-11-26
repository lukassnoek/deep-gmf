#%%
import os
os.environ['DISPLAY'] = ':0.0'
import numpy as np
import pandas as pd
from pathlib import Path
from GFG import Ctx
from GFG.model import Nf
from GFG.identity import IDModel
from itertools import product
from ogbGL.draw_objects import Camera

np.random.seed(42)


def _create_filename(i, gender, ethn, age, xr, yr, zr, xt, yt):
    f_out = f'data/gender-{gender}/ethn-{ethn}/age-{age}/id-{str(i+1).zfill(4)}'
    
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


N_IDS = 1

ctx = Ctx(res=(256, 256), renderscale=4.)
ctx.camera[0] = Camera(ctx.win, ctx._ar,
                       target = [-11.5644, -13.0381, 0],
                       eye = [-11.5644, -13.0381, 450.],
                       up = [0,1,0],
                       FOV = 50,
                       near = 100.0,
                       far = 1000.0)
ctx.transform_lights(0, 0, 0, order='xyz')

data_path = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(data_path)
base_nf = Nf.from_default()

#%%
genders = ['M']
ethns = ['WC']
ages = [25]
rots = [0]#-60, -45, -30, -15, 0, 15, 30, 45, 60]
trans = [0, 30, 90, 100]

n_stim = len(genders) * len(ethns) * len(ages) * \
         len(rots) * 3 * len(trans) * 2

print(f"INFO: generating {n_stim} stimuli!")

#%%
for gender, ethn, age in product(genders, ethns, ages):
    
    for i in range(N_IDS):
        v_coeff = np.random.normal(0, 1, size=len(idm))
        t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))

        nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age, basenf=base_nf)
        center_of_mass = nf.v.mean(axis=0)
        nf.attach(ctx)

        for params in product(rots, rots, rots, trans):
            print(params)
            xr, yr, zr, zt = params
            yt, xt = 0, 0
            f_out = _create_filename(i, gender, ethn, age, xr, yr, zr, xt, yt)

            # Translate to origin, rotate, and apply actual translation
            nf.transform_model(xr, yr, zr, center_of_mass, order='txyzs', replace=True)
            nf.transform_model(0, 0, 0, np.array([xt, yt, zt]), order='t', replace=False)

            # Render + save
            img = ctx.render(dest='image')
            Path(f_out).parent.mkdir(parents=True, exist_ok=True)
            img.save(f_out)
            
        nf.detach()
            
            
                    
# tf.transform_model(x, y, z, tx, ty, tz, scale, order='xyzts')
#ctx.set_lights()
# from ogbGL.draw_objects import Light
# new_light = Light(ctx.win)
# ctx.lights.append(new_light)
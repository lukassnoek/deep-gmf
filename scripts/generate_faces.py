import numpy as np
import pandas as pd
from GFG import Ctx
from GFG.model import Nf
from GFG.identity import IDModel
from itertools import product


val2idx = {
    'rot': {-30: 0, -15: 1, 0: 2, 15: 3, 30: 4},
    'trans': {-10: 0, 0: 1, 10: 2}
}

def _create_filename(i, gender, ethn, age, xr, yr, zr, xt, yt, s):
    f_out = f'id-{str(i+1).zfill(4)}_gender-{gender}_ethn-{ethn}_age-{age}'
    
    for n, p in {'xr': xr, 'yr': yr, 'zr': zr, 'xt': xt, 'yt': yt}.items():
        if isinstance(p, (int, float)):
            if p == 0:
                p = f'n{p}'
            elif p > 0:
                p = f'p{p}'
            else:
                p = f'm{abs(p)}'
                        
        f_out += f'_{n}-{p}'
    
    f_out += f'_s-{str(s).zfill(3)}.png'
    return f_out            


N_IDS = 1

data_path = '/analyse/Project0294/GFG_data/model_HDB_linear_v2dense_compact.mat'
idm = IDModel.load(data_path)

ctx = Ctx(renderscale=4.)

base_nf = Nf.from_default()
genders = ['M', 'F']
ethns = ['WC', 'EA', 'BA']
ages = [15, 25, 35, 45, 55, 65]
rots = [-20, -10, 0, 10, 20]
trans = [-10, 0, 10]
scale = [100, 120, 140, 160, 180]

n_stim = len(genders) * len(ethns) * len(ages) * \
         len(rots) * 3 * len(trans) * 2 * len(scale)
         
print("INFO: generating {n_stim} stimuli!")

for gender, ethn, age in product(genders, ethns, ages):
    
    for i in range(N_IDS):
        v_coeff = np.random.normal(0, 1, size=len(idm))
        t_coeff = np.random.normal(0, 1, size=(idm.nbands, len(idm)))

        nf = idm.generate(v_coeff, t_coeff, ethnicity=ethn, gender=gender, age=age, basenf=base_nf)
        nf.attach(ctx)

        for params in product(rots, rots, rots, trans, trans, trans, scale):
            xr, yr, zr, xt, yt, zt, s = params
            f_out = _create_filename(i, gender, ethn, age, xr, yr, zr, xt, yt, s)
            s /= 100
            nf.transform_model(xr, yr, zr, [xt, yt, zt], s, order='sxyzt', replace=True)
            img = ctx.render(dest='image')
            
            img.save(f'data/{f_out}')
            
        nf.detach()
            
            
                    
# tf.transform_model(x, y, z, tx, ty, tz, scale, order='xyzts')
#ctx.set_lights()
# from ogbGL.draw_objects import Light
# new_light = Light(ctx.win)
# ctx.lights.append(new_light)
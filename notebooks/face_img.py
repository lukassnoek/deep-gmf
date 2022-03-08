import os
# necessary when on deepnet server
os.environ['DISPLAY'] = ':0.0'

from GFG import Ctx
from GFG.model import Nf
from GFG.core import Camera
from GFG.model import Adata

# Load neutral face and animation data (adata)
nf = Nf.from_default()
adata = Adata.load('quick_FACS_blendshapes_v2dense')

# Create rendering context (Ctx)
ctx = Ctx(hidden=True)
nf.attach(ctx)  # attach to openGL context
adata.attach(ctx)

# Custom camera object (not strictly necessary)
ctx._camera[0] = Camera(
    ctx.win, (512, 512), 4.,  # resolution, renderscale
    target=[-11.5644, -13.0381, 0],   # target location
    eye = [-11.5644, -13.0381, 350],  # lens location
    up = [0, 1, 0],
    FOV = 50,
    near = 100.,
    far = 1000.
)
ctx.assign_camera(0)

# Plotting parameters (wt = weight, mx = max, cm = mpl cmap name)
ctx.devmapwt = 1.
ctx.devmapmx = 12.
#ctx.devmapcm('inferno')

# Set amplitude of active AUs
aus = ['AU9', 'AU25']
for au in aus:
    adata.bshapes[au] = 1

img = ctx.render('image')
img.save('AU9+AU25.png')
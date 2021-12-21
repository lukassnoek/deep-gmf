# Function to phase scramble a 2D (RGB[A]) image.
# Adapted from a Matlab script by Martin Hebart
# (http://martin-hebart.de/code/imscramble.m)
# 
# Lukas Snoek, 2021

import warnings
import numpy as np
from PIL import Image
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter


def phase_scramble_image(img_path, out_path, grayscale=True, shuffle_phase=True,
                         smooth=None):
    """ Phase scrambles a 2D (RGB[A]) image. 
    
    Parameters
    ----------
    img_path : str
        Path to image to be scrambled
    out_path : str
        Where to save the phase scrambled image
    grayscale : bool
        Whether to convert the image to grayscale before processing or not
    shuffle_phase : bool
        Whether to shuffle the phases (default) or to offset the phases with
        random numbers between 0-1 (like Martin Hebart's script)
    smooth : int/float
        How much to smooth the phase scrambled image (default: None, no smoothing)          
    """
    
    img = Image.open(img_path)
    if grayscale:
        img = img.convert('L')

    # Rescale to 0-1 range
    img = np.array(img).astype(float)
    img /= 255.

    if img.shape[2] == 4:
        # Remove alpha channel
        img = img[:, :, :3]

    # Add axis if grayscale
    if img.ndim == 2:
        img = img[:, :, None]

    img_scr = np.zeros_like(img)
    x, y, d = img.shape
    
    for dim in range(d):
        freq = fft2(img[:, :, dim])
        amp = np.abs(freq)
        phase = np.angle(freq)
        if shuffle_phase:
            np.random.shuffle(phase)
        else:
            phase += np.angle(fft2(np.random.rand(x, y)))

        with warnings.catch_warnings():
            # Filter ComplexWarning message
            warnings.filterwarnings("ignore", category=np.ComplexWarning)
            img_scr[:, :, dim] = ifft2(amp * np.exp(1j * phase))

    # Cast to real, rescale to 0-1 range, get rid of nan/inf    
    img_scr = np.real(img_scr)
    mn, mx = img_scr.min(axis=(0, 1)), img_scr.max(axis=(0, 1))
    img_scr = (img_scr - mn) / (mx - mn)
    img_scr = np.nan_to_num(img_scr)

    if smooth is not None:
        for i in range(img_scr.shape[2]):
            img_scr[..., i] = gaussian_filter(img_scr[..., i], sigma=smooth)

    # Rescale back to 0-255 range
    img_scr = np.uint8(img_scr * 255).squeeze()
    Image.fromarray(img_scr).save(out_path)


if __name__ == '__main__':
    
    for i in range(3):
        f_out = f'data/background_{i+1}.png'
        phase_scramble_image('test.png', f_out, shuffle_phase=False, grayscale=False)
    
    
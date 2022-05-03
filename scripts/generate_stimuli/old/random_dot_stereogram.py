import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def rds(r=50, shape='square', disparity=5, x=0, y=0, img_size=(128, 128)):
    """ Creates a random dot stereogram.
    
    Parameters
    ----------
    r : int
        Radius in pixels; if shape is `'square'`, then the radius 
        is multiplied with sqrt(pi) to keep the area the same
    shape : str
        Either 'square' or 'circle'
    disparity : int
        Horizontal disparity in number of pixels
    x : int
        Horizontal translation
    y : int
        Vertical translation
    img_size : tuple[int]
        Image size (width, height) in pixels
        
    Returns
    -------
    rds_imgs : tuple
        Left and right RDS images
    """

    if tf.is_tensor(r):
        r = r.numpy()
        
    if tf.is_tensor(shape):
        shape = shape.numpy().decode('utf-8')

    if tf.is_tensor(disparity):
        disparity = disparity.numpy()
        
    if tf.is_tensor(x):
        x = x.numpy()
        
    if tf.is_tensor(y):
        y = y.numpy()

    if r % 2 != 0:
        raise ValueError("`r` should be an even integer!")

    if shape == 'square':
        # Get "radius" of square with equal area of circle
        # with radius `r`
        r = int(round(r * np.sqrt(np.pi)))

    # Define center coordinates of image   
    cx, cy = img_size[0] // 2, img_size[1] // 2

    # Create random dot image
    img_l = np.random.randint(0, 2, size=img_size)

    if shape == 'square':
        mask = np.zeros_like(img_l).astype(bool)
        # x_ and y_ are offset relative to center
        x_, y_ = cx + x, cy + y
        # Set everything inside the shape to True in mask
        mask[x_ - r // 2: x_ + r // 2, y_ - r // 2: y_ + r // 2] = True
    elif shape == 'circle':
        # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
        Y, X = np.ogrid[:img_size[0], :img_size[1]]
        dist_from_center = np.sqrt((X - cx + x) ** 2 + (Y - cy + y) ** 2)
        mask = dist_from_center <= r

    if np.mean(mask) > 0.5:
        raise ValueError("Shape is larger than 80% of the image!")

    # Mask with disparity (= shift horizontally) applied
    mask_disp = np.roll(mask, disparity, axis=1)

    # Create right-eye image
    img_r = img_l.copy()

    # Define "gap" and fill with new values (right
    # side of gap will be overwritten by `mask_disp`)
    gap = np.logical_xor(mask, mask_disp)
    img_r[gap] = np.random.randint(0, 2, size=gap.sum())

    # Set masked values in img_r with values of img_l
    img_r[mask_disp] = img_l[mask]

    # fig, axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(5, 8))
    # axes[0, 0].imshow(img_l, cmap='gray')
    # axes[0, 0].set_title("RDS (left eye)")
    # axes[0, 1].imshow(img_r, cmap='gray')
    # axes[0, 1].set_title("RDS (right eye)")

    # axes[1, 0].imshow(mask, cmap='gray')
    # axes[1, 0].set_title("Shape mask (left eye)")
    # axes[1, 1].imshow(mask_disp, cmap='gray')
    # axes[1, 1].set_title("Shape mask (right eye)")

    # axes[2, 0].imshow(img_l - img_r, cmap='gray')
    # axes[2, 0].set_title("Left img - right img")
    # axes[2, 1].imshow(mask.astype(int) - mask_disp.astype(int), cmap='gray')
    # axes[2, 1].set_title("Left mask - right mask")

    # for ax in axes.flatten():
    #     ax.axis('off')
    # plt.show()

    return img_l[..., None], img_r[..., None] #mask.astype(int)[..., None], mask_disp.astype(int)[..., None]
    
#if __name__ == '__main__':
#    random_dot_stereogram(shape='circle')
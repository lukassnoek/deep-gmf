import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

import cv2
import click
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from pathlib import Path
from skimage.transform import warp, SimilarityTransform


dst = np.array([        
    [43.29459953, 51.69630051],
    [78.53179932, 51.50139999],
    [61.02519989, 71.73660278],
    [46.54930115, 92.36550140],
    [75.72990036, 92.20410156]
])

@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_dir', type=click.Path())
@click.option('--suffix', default='jpg')
@click.option('--depth', default=2)
def main(in_dir, out_dir, suffix='jpg', depth=2):

    detector = MTCNN()
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    imgs = in_dir.glob(f'**/*.{suffix}')
    s_trans = SimilarityTransform()
        
    for img in tqdm(imgs):

        f_out = out_dir / '/'.join(img.parts[-depth:])
        if f_out == img:
            raise ValueError("Overwriting!")
        
        if f_out.is_file():
            continue
    
        img_ = cv2.imread(str(img))[..., ::-1]    
        results = detector.detect_faces(img_)
        
        if not results:
            print(f"Detection failed for: {img}")
            continue

        src = np.array(list(results[0]['keypoints'].values()))
        s_trans.estimate(src, dst)
        img_crop = warp(img_, s_trans.inverse, output_shape=(112, 112), preserve_range=True)
        
        f_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f_out), img_crop.astype(np.uint8)[..., ::-1])


if __name__ == '__main__':
    main()
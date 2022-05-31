import os
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

import cv2
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor
from face_alignment import FaceAlignment, LandmarksType
from skimage.transform import warp, SimilarityTransform


@click.command()
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_dir', type=click.Path())
@click.option('--batch-size', default=32)
@click.option('--depth', default=2)
def main(in_dir, out_dir, batch_size, depth=2):

    dst = np.load('data/lm_template.npy')

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    dataset = ImageFolder(in_dir, transform=PILToTensor())
    n_batches = len(dataset) // batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True, drop_last=False)
    model = FaceAlignment(landmarks_type=LandmarksType._2D, device='cuda')

    failed_files = open(out_dir / 'failed.txt', 'w')
    s_trans = SimilarityTransform()        
    for ib, (X, _) in enumerate(tqdm(loader, total=n_batches)):
        fnames = dataset.imgs[ib*batch_size:(ib+1)*batch_size]
        fnames = [fn[0] for fn in fnames]

        lm_batch, _, box_batch = model.get_landmarks_from_batch(X, return_bboxes=True)
        for i, lm in enumerate(lm_batch):
            if isinstance(lm, list):
                print(f"{fnames[i]}: no face detected")
                failed_files.write(f'{fnames[i]}\tdetectionfail\n')
                continue
                      
            if lm.shape[0] != 68:
                nf = lm.shape[0] // 68
                #print(f"{fnames[i]}: {nf} faces detected")
                most_conf = np.argmax([b[4] for b in box_batch[i]])
                lm = lm[most_conf*68:(most_conf+1)*68, :]
                failed_files.write(f'{fnames[i]}\t{nf}faces\n')

            s_trans.estimate(lm, dst)
            img = X[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
            img_crop = warp(img, s_trans.inverse, output_shape=(112, 112), preserve_range=True)
        
            f_out = Path(fnames[i])
            f_out = out_dir / '/'.join(f_out.parts[-depth:])
            f_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(f_out), img_crop.astype(np.uint8)[..., ::-1])

    failed_files.close()

if __name__ == '__main__':
    main()
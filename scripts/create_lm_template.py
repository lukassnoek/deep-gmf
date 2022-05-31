import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor
from face_alignment import FaceAlignment, LandmarksType

batch_size = 32
total = 2048
dataset = ImageFolder('/analyse/Project0257/lukas/data/images', transform=PILToTensor())
n_batches = total // batch_size
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
model = FaceAlignment(landmarks_type=LandmarksType._2D, device='cuda')

template = np.zeros((68, 2))
counter = 0
for ib, (X, _) in enumerate(tqdm(loader, total=n_batches)):
    fnames = dataset.imgs[ib*batch_size:(ib+1)*batch_size]
    fnames = [fn[0] for fn in fnames]
    lm_batch = model.get_landmarks_from_batch(X)
    valid = []
    for lm in lm_batch:
        if isinstance(lm, list):
            continue
        
        if lm.shape[0] != 68:
            continue

        valid.append(lm)    
    
    if len(valid) == 0:
        continue
    
    template += np.stack(valid, axis=0).mean(axis=0)
    counter += 1
    
    if (counter * batch_size) >= total:
        break

template /= counter
template = np.round(template).astype(np.uint8)

image = np.zeros((112, 112))
for i in range(68):
    cv2.circle(image, template[i, :], radius=1, color=(255, 0, 0), thickness=-1)

cv2.imwrite('data/lm_template.png', image)
np.save('data/lm_template.npy', template)
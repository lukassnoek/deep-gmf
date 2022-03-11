# Training notes

Notes on traning CNNs on the data from Daube et al. (2021).

### 5 Oct 2021

* *PikeNet* (no pooling, only conv downsampling) works just fine with few filters
* Adam optimizer should use a learning rate >= 0.001 (otherwise overshoot)
* Both *PikeNet* and *CORnet_Z* get >50% accuracy after 2 epochs, but the former trains way quicker
* *ResNet{6,10}* overfits like crazy.
  * Seems that BatchNorm is messing up things
  * Problem is the default momentum parameter (0.99); because of no bias correction?
  * Setting momentum lower (e.g., 0.5) helps, and smaller batch size too ( = more steps, more updating)
  * "Optimal parameters" (val accuracy > 0.9 after three epochs of 2^14 = 16384 samples): bn_momentum = 0.1, batch size = 64

### 30 Dec 2021

* Distributed training is slower (and less accurate) than single-GPU ...
* Small batch sizes (e.g. 256) are both quicker and more accurate on single GPU

### 2 Mar 2022
Messing around with training models that predict shape (even just a single shape parameter) and trying to figure out how much data (separate identities and variations per identity) you need in order to train it to a reasonable degree of accuracy. A couple of observations:

* You definitely need more than 1 variation per identity!
* The model will, at some point, overfit to the training set, i.e., validation loss will halt and then increase at some point, but training loss will keep decreasing
* Overfitting can, to some degree, be countered by training on more identities. E.g., when training on 1024 identities, validation cosine similarity will plateau at 0.5, training on 2048 identities will plateau at 0.6, etc. 

### 11 Mar 2022
Experimenting with how many variations per ID you need to train a simple ResNet10 to classify ID. Weirdly, fewer variations is possible when setting the learning rate a lot higher. So far, it works for 32 variations per ID with a learning rate of 0.005 (not higher!).
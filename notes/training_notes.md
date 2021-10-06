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
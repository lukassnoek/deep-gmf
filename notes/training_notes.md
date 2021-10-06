# Training notes

Notes on traning CNNs on the data from Daube et al. (2021).

### 5 Oct 2021

* *PikeNet* (no pooling, only conv downsampling) works just fine with few filters
* Adam optimizer should use a learning rate >= 0.001 (otherwise overshoot)
* Both *PikeNet* and *CORnet_Z* get >50% accuracy after 2 epochs, but the former trains way quicker
* *ResNet{6,10}* overfit like crazy.
  * Seems that BatchNorm is messing up things
  * Turning off BatchNorm fixes the issue, but leads to slower learning ...
  * What doesn't work:
    * increasing batch size, validation set size
    * setting `center` and `scale` args to false
    * changing conv initializer to default
  * Idea: add batch norm to skip layer?
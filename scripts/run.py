import sys
sys.path.append('.')
from src.models import ResNet6, CORnet_Z, PikeNet
from src.utils.io import create_data_generator
from tensorflow.keras.optimizers import Adam

model = ResNet6(n_classes=4, bn_momentum=0.1)
opt = Adam(learning_rate=0.001)  # lr > 0.001 leads to overshoot
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

train_gen, val_gen = create_data_generator('data/human_exp/dataset_info.csv', 'face_id',
                                           n=16384, n_validation=2048, batch_size=256)
model.fit(train_gen, validation_data=val_gen, epochs=100)
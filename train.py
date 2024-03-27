#https://github.com/AIKONG2024/AFD-MultiScale-Net

import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import *
import pandas as pd
import threading
import random
import rasterio
import os
import numpy as np
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import joblib
import time
from models import AFD_MultiScale_Net
from metrics import miou

#random seed
RANDOM_STATE = 42 # seed 고정
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

MAX_PIXEL_VALUE = 65535 # nomalization picxel numbers

N_FILTERS = 16 # 논문 구현 내용에서 filter 개수 16 고정
N_CHANNELS = 4 # 채널 swir2, nir, blue, afi 4개 고정
EPOCHS = 200 # 훈련 epoch 200 고정
BATCH_SIZE = 15 # 배치 15 고정
IMAGE_SIZE = (256, 256) # 이미지 크기 256 x 256 고정
MODEL_NAME = 'AFD_MultiScale_Net' # 모델 이름
THESHOLDS = 0.25

# name - 중복되지 않게
import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr

# data path
IMAGES_PATH = './datasets/train_img/'
MASKS_PATH = './datasets/train_mask/'

# save weights path
OUTPUT_DIR = f'./datasets/train_output/{save_name}/'
WORKERS = 32

# final weights path
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

# Cuda (gpu 개수 1개)
CUDA_DEVICE = 0

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    with rasterio.open(path) as src:
        swir2 = src.read(7).astype(float)
        swir1 = src.read(6).astype(float)
        nir = src.read(5).astype(float)
        blue = src.read(2).astype(float)

    swir2 = swir2 / MAX_PIXEL_VALUE
    swir1 = swir1 / MAX_PIXEL_VALUE
    nir = nir / MAX_PIXEL_VALUE
    blue = blue / MAX_PIXEL_VALUE

    # AFI
    afi = np.divide(swir2, blue, out=np.zeros_like(swir2), where=blue!=0)
    afi_min = np.min(afi)
    afi_max = np.max(afi)
    afi = (afi - afi_min) / (afi_max - afi_min)

    img = np.stack([swir2, nir, blue, afi], axis=-1)
    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

def shuffle_lists(images_path, masks_path, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    combined = list(zip(images_path, masks_path))
    np.random.shuffle(combined)
    shuffled_images_path, shuffled_masks_path = zip(*combined)
    return list(shuffled_images_path), list(shuffled_masks_path)

def augment_image(image, mask, per=0.4):
    #Image augmentation 위 아래 flip 증폭 
    if random.random() < per:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        
    if random.random() < per:
        image = np.flipud(image)
        mask = np.flipud(mask)

    return image, mask

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, is_train = False):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr
        
    i = 0
    # data shuffle
    while True:

        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            
            if is_train:
                img, mask = augment_image(img, mask)
                
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

# GPU setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

train_meta = pd.read_csv('./datasets/train_meta.csv')
test_meta = pd.read_csv('./datasets/test_meta.csv')

# create folder when not exist folder
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#data
x_tr, x_val = train_test_split(train_meta, test_size=0.3, random_state=RANDOM_STATE)

images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, is_train=True)
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)

#model
model = AFD_MultiScale_Net(input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS, nClasses=1, dilation_rate=2)
model.compile(optimizer = Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics = ['accuracy', miou])
model.summary()

print('[start train]')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    epochs=EPOCHS,
    workers=WORKERS
)
print('[end train]')

#save weights
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("saved weight file name: {}".format(model_weights_output))
y_pred_dict = {}

for idx, i in enumerate(test_meta['test_img']):
    img = get_img_arr(f'./datasets/test_img/{i}') 
    y_pred = model.predict(np.array([img]), batch_size=32)
    y_pred = np.where(y_pred[0, :, :, 0] > THESHOLDS, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

#save predict pkl
joblib.dump(y_pred_dict, f'./predict/{MODEL_NAME}_{save_name}_y_pred.pkl')
print("saved pkl:", f'./predict/{MODEL_NAME}_{save_name}_y_pred.pkl')

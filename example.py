from keras.optimizers import *
from models import AFD_MultiScale_Net
from metrics import miou
import tensorflow as tf

# model 사용
N_FILTERS = 16 # filter
N_CHANNELS = 3 # channel 
BATCH_SIZE = 16 # batch size 
IMAGE_SIZE = (256, 256) # image size

#AFD_MultiScale_Net 사용
model = AFD_MultiScale_Net(input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS, nClasses=1, dilation_rate=2)
model.compile(optimizer = Adam(learning_rate=1e-2), loss='binary_crossentropy', metrics = ['accuracy', miou])
model.summary()
from keras.models import *
from keras.layers import *


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, dilation_rate=1):
    
    '''
    - conv2D block
    L1 : Conv2D(3,3) + ReLU + variableDir(dilation_rate(1))
    L2 : Conv2D(3,3) + ReLU + variableDir(dilation_rate(1 or 2 or 3)) 
    L1 + L2 (concatenate)
    '''
    
    #L1
    c1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)
    if batchnorm:
        c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)

    #L2
    c2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), dilation_rate=dilation_rate, padding="same")(input_tensor)
    if batchnorm:
        c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)

    return Concatenate()([c1, c2])

def upsample_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True):
    '''
    - upsample block
    Conv2DTranspose(3,3) + BatchNormalization + ReLU 
    '''
    x = Conv2DTranspose(n_filters, kernel_size, strides=(2, 2), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = UpSampling2D(x)
    x = Concatenate()([x, skip_tensor])
    return x

def conv2d_sets(input_tensor, n_filters, batchnorm, dilation_rate, k = 357):
    '''
    - conv2D block sets
    conv2D 3x3, 5x5, 7x7 block set
    k3 : 3x3 / k35 : 3,3 + 5,5 / k357 : 3,3 + 5,5 + 7,7
    '''
    l_arr = []
    if k == 3 : 
        c1 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm, dilation_rate=dilation_rate)
        l_arr.append(c1)
    elif k == 35:
        c1 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm, dilation_rate=dilation_rate)
        c2 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=5, batchnorm=batchnorm, dilation_rate=dilation_rate)
        l_arr.append(c1)
        l_arr.append(c2)
    elif k == 357:
        c1 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm, dilation_rate=dilation_rate)
        c2 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=5, batchnorm=batchnorm, dilation_rate=dilation_rate)
        c3 = conv2d_block(input_tensor, n_filters=n_filters, kernel_size=7, batchnorm=batchnorm, dilation_rate=dilation_rate)
        l_arr.append(c1)
        l_arr.append(c2)
        l_arr.append(c3)
    return Concatenate()(l_arr)

def AFD_MultiScale_Net(input_height=256, input_width=256, nClasses=1, n_filters=16, batchnorm=True, n_channels=1, dilation_rate = 1):
    input_img = Input(shape=(input_height, input_width, n_channels))
    
    # 인코더
    e1 = conv2d_sets(input_img, n_filters*1, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p1 = MaxPooling2D((2, 2))(e1)
    
    e2 = conv2d_sets(p1, n_filters*2, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p2 = MaxPooling2D((2, 2))(e2)
    
    e3 = conv2d_sets(p2, n_filters*4, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p3 = MaxPooling2D((2, 2))(e3)
    
    e4 = conv2d_sets(p3, n_filters*8, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p4 = MaxPooling2D((2, 2))(e4)
    
    e5 = conv2d_sets(p4, n_filters*16, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    # 디코더
    d1 = upsample_block(e5, e4, n_filters*8, kernel_size=3, batchnorm=batchnorm)
    d2 = conv2d_sets(d1, n_filters*8, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d2 = upsample_block(d2, e3, n_filters*4, kernel_size=3, batchnorm=batchnorm)
    d3 = conv2d_sets(d2, n_filters*4, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d3 = upsample_block(d3, e2, n_filters*2, kernel_size=3, batchnorm=batchnorm)
    d4 = conv2d_sets(d3, n_filters*2, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    d4 = upsample_block(d4, e1, n_filters, kernel_size=3, batchnorm=batchnorm)
    d5 = conv2d_sets(d4, n_filters, batchnorm=batchnorm, dilation_rate=dilation_rate)

    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(d5)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
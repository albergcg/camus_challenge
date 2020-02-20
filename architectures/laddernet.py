# Implementation of LadderNet using the tf.keras Functional API

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

drop = 0.25

def ResBlock(input_tensor, filters):
    """"
    Define Residual Block showed in Zhuang (2019)
    CONV => BATCH => RELU => DROPOUT => CONV => BATCH
      |                                   |
      |----------- SHARED LAYER ----------|
      
    """
        
    conv_1 = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')    
    conv_1a = conv_1(input_tensor) # Shared weights conv layer
    batch_1 = BatchNormalization()(conv_1a)
    relu_1  = Activation("relu")(batch_1)
    drop_1  = Dropout(drop)(relu_1)
    conv_1b = conv_1(drop_1) # Shared weights conv layer
    batch_1 = BatchNormalization()(conv_1b)
    return batch_1

def LadderNet(input_size = (256, 256, 1), num_classes=2, filters=30):
    
    """
    LadderNet (Zhuang, 2019) implementation in tensorflow.keras
    Method: Keras Functional API
    """  
    
    # X's denote standard flow
    # XNUM denote ResBlock outputs
    
    # "First" UNet
    
    # Input branch
    inputs = Input(input_size)
    X = Conv2D(filters=filters, kernel_size=3, activation="relu", padding = 'same', kernel_initializer = 'he_normal')(inputs)

    # Down branch
    X1 = ResBlock(input_tensor=X, filters=filters) # ResBlock located in the first layer of the paper scheme
    X = Conv2D(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X1) 
    X = Activation("relu")(X) # This ReLU is not shown in the paper scheme
    
    X2 = ResBlock(input_tensor=X, filters=filters*2)
    X = Conv2D(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X2)
    X = Activation("relu")(X)
    
    X3 = ResBlock(input_tensor=X, filters=filters*4)
    X = Conv2D(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X3)
    X = Activation("relu")(X)
    
    X4 = ResBlock(input_tensor=X, filters=filters*8)
    X = Conv2D(filters=filters*16, kernel_size=3, strides=2, kernel_initializer='he_normal')(X4)
    X = Activation("relu")(X)
    
    # Bottom block 
    X = ResBlock(input_tensor=X, filters=filters*16)
    
    # Up branch
    X = Conv2DTranspose(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X4])
    # X = Activation("relu")(X) # This ReLU is commented in the paper code
    X5 = ResBlock(input_tensor=X, filters=filters*8)
    
    X = Conv2DTranspose(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X5)
    X = Add()([X, X3])
    # X = Activation("relu")(X)
    X6 = ResBlock(input_tensor=X, filters=filters*4)
    
    X = Conv2DTranspose(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X6)
    X = Add()([X, X2])
    # X = Activation("relu")(X)
    X7 = ResBlock(input_tensor=X, filters=filters*2)
        
    X = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, output_padding=1, kernel_initializer='he_normal')(X7)
    X = Add()([X, X1])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters)
    
    # Top block (bottle-neck)
    X8 = ResBlock(input_tensor=X, filters=filters)
    X = ResBlock(input_tensor=X, filters=filters)
    X = Add()([X, X8])
    
    # "Second" UNet
    
    # Down branch
    X9 = ResBlock(input_tensor=X, filters=filters)
    X = Conv2D(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X) 
    X = Activation("relu")(X)
    X = Add()([X7, X])    
    
    X10 = ResBlock(input_tensor=X, filters=filters*2)
    X = Conv2D(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)    
    X = Activation("relu")(X)    
    X = Add()([X6, X])
    
    X11 = ResBlock(input_tensor=X, filters=filters*4)
    X = Conv2D(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)    
    X = Activation("relu")(X)
    X = Add()([X5, X])

    X12 = ResBlock(input_tensor=X, filters=filters*8)
    X = Conv2D(filters=filters*16, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)    
    X = Activation("relu")(X)
    
    # Bottom block
    X = ResBlock(input_tensor=X, filters=filters*16)
    
    # Up branch
    X = Conv2DTranspose(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X12])   
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*8)
    
    X = Conv2DTranspose(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X11])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*4)
    
    X = Conv2DTranspose(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X10])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*2)
    
    X = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, kernel_initializer='he_normal', output_padding=1)(X)
    X = Add()([X, X9])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters)
    
    # Final block
    X = Conv2D(filters=num_classes, kernel_size=1, kernel_initializer='he_normal')(X)
    # X = Activation("relu")(X)
    X = Activation("softmax")(X)
    #X = Conv2D(1, 1)(X)
    
    model = Model(inputs, X)
    
    
    return model
    


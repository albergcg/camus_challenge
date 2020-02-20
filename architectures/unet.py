from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def conv2D_module(inputs, filters, kernel_size=3, padding="valid", pool_size=2):
    
    """
    CONV => RELU => CONV => RELU => MAXPOOL
    """
    
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
               kernel_initializer='he_normal')(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
               kernel_initializer='he_normal')(inputs)
    x = Activation("relu")(x)


def UNet(input_size, depth, num_classes, filters, batch_norm):
    
    """
    UNet (Ronneberger, 2015) implementation in tensorflow.keras
    using Keras Functional API.
    """
    
    # Input layer
    inputs = Input(input_size)
    x = inputs
    
    # Encoding
    down_list = []
    for layer in range(depth):
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        
        if batch_norm: 
            x = BatchNormalization()(x)
            x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
            x_down = BatchNormalization()(x)
        else:
            x_down = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        
        down_list.append(x_down)
        x = MaxPooling2D(pool_size=2)(x_down)
        filters = filters*2
    
    # Bottom
    x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if batch_norm: x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if batch_norm: x = BatchNormalization()(x)
    
    # Decoding
    for layer in reversed(down_list):
        filters = filters // 2
        x = UpSampling2D((2,2))(x)
        x = concatenate([x, layer])
        x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
        if batch_norm: x = BatchNormalization()(x)
    
    # Output layer
    x = Conv2D(filters=num_classes, kernel_size=1)(x)
    if batch_norm: x = BatchNormalization()(x)
    outputs = Activation("softmax")(x)
    
    model = Model(inputs, outputs)
    return model
    
    """
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    """
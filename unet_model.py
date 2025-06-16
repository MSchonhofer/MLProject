from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose,
                                     concatenate, Cropping2D, ZeroPadding2D,
                                     BatchNormalization, Activation)
from tensorflow.keras.models import Model

def build_unet_model(base_filters=32, input_shape=(308, 384, 1)):
    inputs = Input(input_shape)
    padded = ZeroPadding2D(((2, 2), (0, 0)))(inputs)

    # Encoder
    c1 = Conv2D(base_filters, 3, padding='same', activation='relu')(padded)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(base_filters, 3, padding='same', activation='relu')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(base_filters * 2, 3, padding='same', activation='relu')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(base_filters * 2, 3, padding='same', activation='relu')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(base_filters * 4, 3, padding='same', activation='relu')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(base_filters * 4, 3, padding='same', activation='relu')(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    b = Conv2D(base_filters * 8, 3, padding='same', activation='relu')(p3)
    b = BatchNormalization()(b)
    b = Conv2D(base_filters * 8, 3, padding='same', activation='relu')(b)

    # Decoder
    u1 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding='same')(b)
    u1 = concatenate([u1, c3])
    u1 = Conv2D(base_filters * 4, 3, padding='same', activation='relu')(u1)
    u1 = BatchNormalization()(u1)

    u2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding='same')(u1)
    u2 = concatenate([u2, c2])
    u2 = Conv2D(base_filters * 2, 3, padding='same', activation='relu')(u2)
    u2 = BatchNormalization()(u2)

    u3 = Conv2DTranspose(base_filters, 2, strides=2, padding='same')(u2)
    u3 = concatenate([u3, c1])
    u3 = Conv2D(base_filters, 3, padding='same', activation='relu')(u3)
    u3 = BatchNormalization()(u3)

    out = Conv2D(1, 1, activation='sigmoid', dtype='float32')(u3)
    out = Cropping2D(((2, 2), (0, 0)))(out)

    return Model(inputs, out)
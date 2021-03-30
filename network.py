from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, concatenate, Lambda, add, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers

is_batch_norm = True
is_drop_out = True

''' U-Net '''
def build_unet_model(batch_shape,
                    ch_num,
                    drop_rate=0.1,
                    transfer_learn=False,
                    transfer_encoder=False,
                    lr=0.001
                    ):
    def encode_block(x, ch):
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            return x
        
        x = base_block(x)
        x = base_block(x)
        return x
    
    def decode_block(x, c, ch):
        ch = ch
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        x = base_block(x)
        x = base_block(x)
        return x
    
    input_batch = Input(shape=(*batch_shape, ch_num))
    e0 = Conv2D(8, (1, 1), padding='same')(input_batch)
    e0 = Activation('relu')(e0)

    e0 = encode_block(e0, 16)

    e1 = AveragePooling2D((2, 2))(e0)
    e1 = encode_block(e1, 32)

    e2 = AveragePooling2D((2, 2))(e1)
    e2 = encode_block(e2, 64)

    e3 = AveragePooling2D((2, 2))(e2)
    e3 = encode_block(e3, 128)

    d2 = decode_block(e3, e2, 64)
    d1 = decode_block(d2, e1, 32)
    d0 = decode_block(d1, e0, 16)

    # d0 = Conv2D(2, (1, 1), padding='same')(d0)
    # output_batch = Activation('tanh')(d0)
    # output_batch = Conv2D(2, (1, 1), padding='same')(d0)
    output_batch = Conv2D(3, (1, 1), padding='same')(d0)

    model = Model(input_batch, output_batch)

    # Transfer Learning
    if transfer_learn:
        for l in model.layers[:38]:
            l.trainable = False
    elif transfer_encoder:
        for l in model.layers[38:]:
            l.trainable = False

    # adam = optimizers.Adam(lr=lr, decay=decay)
    adam = optimizers.Adam(lr=lr)
    model.compile(
                # optimizer='adam',
                optimizer=adam,
                metrics=['accuracy'],
                # loss='mean_squared_error'
                loss='mean_absolute_error'
                )
    return model
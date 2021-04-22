from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, concatenate, Lambda, add, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers
from keras.applications.vgg16 import VGG16

is_batch_norm = True
is_dropout = True
# is_dropout = False

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
            # x = Dropout(rate=drop_rate)(x)
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
            # x = Dropout(rate=drop_rate)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        x = base_block(x)
        x = base_block(x)
        return x

    def mean_squared_error_masked(y_true, y_pred):
        print('y_true: ', y_true)
        print('y_pred: ', y_pred)
        gt = y_true[:, :, :, :3]
        mask = y_true[:, :, :, 3:]

        mask = K.cast(mask, 'float32')
        length = K.sum(mask)
        mse = K.sum(K.square(gt - y_pred) * mask) / length
        # mse = K.sum(K.abs(gt - y_pred) * mask) / length
        return mse
    
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
    output_batch = Conv2D(ch_num, (1, 1), padding='same')(d0)

    model = Model(input_batch, output_batch)

    # Transfer Learning
    if transfer_learn:
        for layer in model.layers[:38]:
            layer.trainable = False
    elif transfer_encoder:
        for layer in model.layers[38:]:
            layer.trainable = False

    # adam = optimizers.Adam(lr=lr, decay=decay)
    adam = optimizers.Adam(lr=lr)
    model.compile(
                # optimizer='adam',
                optimizer=adam,
                metrics=['accuracy'],
                # loss='mean_squared_error'
                # loss='mean_absolute_error'
                loss=mean_squared_error_masked
                )
    return model

# Loss Model using Pre-trained VGG-16
def build_loss_model(batch_shape, ch_num):
    input_batch = Input(shape=(*batch_shape, ch_num))
    output_input = Activation('relu')(input_batch)

    vgg_model = VGG16(weights='imagenet', 
                    include_top=False, 
                    input_tensor=input_batch,
                    input_shape=(*batch_shape, ch_num))
    vgg_model.trainable = False

    selected_layers = [1,2,9,10,17,18]
    selected_layers = [2,10,18]
    selected_outputs = [vgg_model.layers[i].output for i in selected_layers]
    # print(selected_outputs)
    # selected_outputs.append(vgg_model.input)
    selected_outputs.append(output_input)
    # print(selected_outputs)
    # loss_model = Model(vgg_model.input, selected_outputs)
    loss_model = Model(input_batch, selected_outputs)
    # loss_model = Model(vgg_model.input, vgg_model.output)
    # loss_model_outputs = loss_model(model.output)
    for layer in loss_model.layers:
        layer.trainable = False
    loss_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # print('loss model buit')
    return loss_model

# U-net with Perceptual Loss using Pre-trained VGG-16
def build_unet_percuptual_model(
                    batch_shape,
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
            if is_dropout:
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
            if is_dropout:
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
    d0 = Conv2D(8, (3, 3), padding='same')(d0)

    # d0 = Conv2D(2, (1, 1), padding='same')(d0)
    # output_batch = Activation('tanh')(d0)
    # output_batch = Conv2D(2, (1, 1), padding='same')(d0)
    output_batch = Conv2D(ch_num, (1, 1), padding='same')(d0)

    model = Model(input_batch, output_batch)
    # print('main model buit')

    loss_model = build_loss_model(batch_shape, ch_num)
    loss_model_outputs = loss_model(model.output)

    full_model = Model(model.input, loss_model_outputs)

    def perceptual_loss(y_true, y_pred):
        # print('y_true: ', y_true)
        # print('y_pred: ', y_pred)
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(*batch_shape, ch_num))
        vgg_model.trainable = False
        selected_layers = [1,2,9,10,17,18]
        selected_outputs = [vgg_model.layers[i].output for i in selected_layers]
        # selected_outputs.append(model.inputs)
        loss_model = Model(vgg_model.input, selected_outputs)
        loss_model_outputs = loss_model(model.output)
        loss_model.compile(optimizer='adam', loss='mse')   
        print('loss model buit')

        f_true = loss_model.predict(y_true)
        f_pred = loss_model.predict(y_pred)
        loss = K.mean(K.square(y_true - y_pred)) #+ K.mean(K.square(f_true - f_pred))
        return loss

    # full_model = Model(model.inputs, loss_model.outputs)


    # Transfer Learning
    if transfer_learn:
        for layer in model.layers[:38]:
            layer.trainable = False
    elif transfer_encoder:
        for layer in model.layers[38:]:
            layer.trainable = False

    # adam = optimizers.Adam(lr=lr, decay=decay)
    adam = optimizers.Adam(lr=lr)
    full_model.compile(
                # optimizer='adam',
                optimizer=adam,
                metrics=['accuracy'],
                loss='mean_squared_error'
                # loss='mean_absolute_error'
                # loss=mean_squared_error_masked
                # loss=perceptual_loss
                )
    return full_model
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from util import *
import config
import tensorflow as tf

########################################################################
# MODEL
########################################################################


def discriminator(fm):
    f0 = UpSampling2D(size=(2, 2), name='dis_up0')(fm[3])
    m1 = concatenate([f0, fm[2]], axis=3)
    f1 = UpSampling2D(size=(2, 2), name='dis_up1')(m1)
    m2 = concatenate([f1, fm[1]], axis=3)
    f2 = UpSampling2D(size=(2, 2), name='dis_up2')(m2)
    m3 = concatenate([f2, fm[0]], axis=3)
    m3 = Dropout(0.5)(m3)
    x = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal', name='dis_conv1')(m3)
    x = BatchNormalization(name='dis_bn1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='dis_pooling1')(x)
    x = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=(2, 2),
               kernel_initializer='he_normal', name='dis_conv2')(x)
    x = BatchNormalization(name='dis_bn2')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='dis_pooling2')(x)
    x = Conv2D(32, 1, activation='relu', padding='same', dilation_rate=(4, 4),
               kernel_initializer='he_normal', name='dis_conv3')(x)
    x = BatchNormalization(name='dis_bn3')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='dis_pooling3')(x)
    x = Conv2D(1, 1, activation='relu', padding='same', dilation_rate=(8, 8),
               kernel_initializer='he_normal', name='dis_conv4')(x)
    x = BatchNormalization(name='dis_bn4')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dis_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='dis_dense2')(x)
    return x


def res_block(x, nb_filters, strides, key):
    res_path = BatchNormalization(name=key+"_bn1")(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0],
                      name=key+"_conv1")(res_path)
    res_path = Dropout(0.3)(res_path)
    res_path = BatchNormalization(name=key+"_bn2")(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1],
                      name=key+"_conv2")(res_path)
    res_path = Dropout(0.3)(res_path)

    shortcut = Conv2D(filters=nb_filters[1], kernel_size=(1, 1), strides=strides[0],
                      name=key+"_conv3")(x)
    shortcut = Dropout(0.5)(shortcut)
    shortcut = BatchNormalization(name=key+"_bn3")(shortcut)
    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1),
                       name='seg_encoder_conv1')(x)
    main_path = BatchNormalization(name='seg_encoder_bn1')(main_path)
    main_path = Activation(activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1),
                       name='seg_encoder_conv2')(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                      name='seg_encoder_conv3')(x)
    shortcut = BatchNormalization(name='seg_encoder_bn2')(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)], 'seg_encoder1')
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)], 'seg_encoder2')
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2), name='seg_up1')(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)], 'seg_decoder1')

    main_path = UpSampling2D(size=(2, 2), name='seg_up2')(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)], 'seg_decoder2')

    main_path = UpSampling2D(size=(2, 2), name='seg_up3')(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)], 'seg_decoder3')

    return main_path


def res_unet(image):
    to_decoder = encoder(image)
    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)], 'seg_path')
    to_decoder.append(path)
    path = decoder(path, from_encoder=to_decoder)
    path = Dropout(0.5)(path)
    mask = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid',
                  name='seg_conv')(path)
    return mask, to_decoder


def build(mode, config):
    input_image = Input(shape=(config.IMAGE_SHAPE, config.IMAGE_SHAPE, 1), name='input_image')
    input_mask = Input(shape=(config.IMAGE_SHAPE, config.IMAGE_SHAPE, 1), name='input_mask')
    domain_cls = Input(shape=[1], name="domain_cls")
    segmentation, fm = res_unet(input_image)
    validity = discriminator(fm)

    if mode == 'trainS' or mode == 'detect':
        seg_loss = Lambda(lambda x: seg_loss_graph(*x), name="segLoss")([input_mask, segmentation])
        dsc = Lambda(lambda x: dice_coef(*x), name="DSC")([input_mask, segmentation])
        acc = Lambda(lambda x: getAcc(*x), name="Acc")([input_mask, segmentation])
        precision = Lambda(lambda x: getPrecision(*x), name="Precision")([input_mask, segmentation])
        recall = Lambda(lambda x: getRecall(*x), name="Recall")([input_mask, segmentation])
        loss_names = ["segLoss"]
        model = Model(inputs=[input_image, input_mask], outputs=[segmentation, seg_loss, dsc, acc, precision, recall])
    elif mode == 'trainD':
        dom_loss = Lambda(lambda x: dom_loss_graph(*x), name="domLoss")([domain_cls, validity])
        acc = Lambda(lambda x: getAcc(*x), name="Acc")([domain_cls, validity])
        loss_names = ["domLoss"]
        model = Model(inputs=[input_image, domain_cls], outputs=[validity, dom_loss, acc, segmentation])
    elif mode == 'trainSS':
        adv_loss = Lambda(lambda x: adv_loss_graph(*x), name="advLoss")(
            [input_mask, domain_cls, segmentation, validity])
        # adv_loss = Lambda(lambda x: dom_loss_graph(*x), name="advLoss")([domain_cls, validity])
        dsc = Lambda(lambda x: dice_coef(*x), name="DSC")([input_mask, segmentation])
        acc = Lambda(lambda x: getAcc(*x), name="Acc")([domain_cls, validity])
        loss_names = ["advLoss"]
        model = Model(inputs=[input_image, input_mask, domain_cls], outputs=[segmentation, adv_loss, dsc, acc])
    else:
        raise SystemExit('Please re-enter mode')

    model._losses = []
    model._per_input_losses = {}
    for loss_name in loss_names:
        layer = model.get_layer(loss_name)
        if layer.output in model.losses:
            continue
        loss = tf.reduce_mean(layer.output, keepdims=True)
        model.add_loss(loss)

    # TODO:添加 multi-GPU
    return model


def train(model, train_dataset, val_dataset, mode, log_dir, checkpoint_path,
          epoch, epochs, learning_rate=config.LRARNING_RATE, momentum=config.MOMENTUM,):
    # Data generator
    train_generator = data_generator(train_dataset, mode, batch_size=config.BATCH_SIZE)
    val_generator = data_generator(val_dataset, mode, batch_size=config.BATCH_SIZE)

    """
    # Callbacks
    """
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    # reduce_lr = ReduceLROnPlateau(monitor='val_seg_conv_dice_coef', patience=10, mode='auto')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    _, full_name = os.path.split(log_dir)
    csv_name, _ = os.path.splitext(full_name)
    csv_name = csv_name + '.csv'
    csv_log = os.path.join(config.METRIC, csv_name)
    csv_logger = CSVLogger(csv_log)
    callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                 ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True),
                 reduce_lr,
                 csv_logger]

    # TODO:不同的mode
    # S_optimizer:adam, D_optimizer:SGD
    if mode == 'trainS':
        metric_names = ["DSC", "Acc", "Precision", "Recall"]
        layer_regex = r"(seg\_.*)"
        for layer in model.layers:
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            layer.trainable = trainable
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        model.compile(optimizer, loss=[None]*len(model.outputs))
    elif mode == 'trainD':
        metric_names = ['Acc']
        layer_regex = r"(dis\_.*)"
        for layer in model.layers:
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            layer.trainable = trainable
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        model.compile(optimizer, loss=[None]*len(model.outputs))
    else:
        metric_names = ["DSC", "Acc"]
        layer_regex = r"(seg\_.*)"
        for layer in model.layers:
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            layer.trainable = trainable
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8)
        model.compile(optimizer, loss=[None]*len(model.outputs))

    for metric_name in metric_names:
        if metric_name in model.metrics_names:
            continue
        layer = model.get_layer(metric_name)
        model.metrics_names.append(metric_name)
        metric_val = tf.reduce_mean(layer.output)
        model.metrics_tensors.append(metric_val)

    if epoch >= epochs:
        raise Exception("epoch must smaller than epochs!")

    model.fit_generator(
        train_generator,
        initial_epoch=epoch,
        epochs=epochs,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=config.VALIDATION_STEPS,
    )


########################################################################
# TRAIN and TEST
########################################################################


def segment(model, image_path):
    print("Running on {}".format(image_path))
    img = skimage.io.imread(image_path)
    img = reshape(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.expand_dims(img, axis=0)
    mask = model.predict(img, verbose=0)
    return mask

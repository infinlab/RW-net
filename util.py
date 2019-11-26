import config
import os
import numpy as np
import skimage.io
from keras import backend as K
import datetime
import re
from keras.metrics import binary_accuracy
import tensorflow as tf
########################################################################
# Dataset
########################################################################


def data_prepare(dataset_dir, subset, folder):
    """

    :param dataset_dir:
    :param subset:
    :return: [{'ids': image, 'cls': 1},
              { }]
    """
    data_meta = []
    if folder == 'real':
        folder_path = os.path.join(dataset_dir, subset, folder)
        dataset_path = os.path.join(folder_path, 'image')
        images = next(os.walk(dataset_path))[2]
        for image in images:
            data_meta.append({'path': folder_path, 'ids': image, 'cls': 1})
    elif folder == 'fake':
        folder_path = os.path.join(dataset_dir, subset, folder)
        dataset_path = os.path.join(folder_path, 'image')
        images = next(os.walk(dataset_path))[2]
        for image in images:
            data_meta.append({'path': folder_path, 'ids': image, 'cls': 0})
    else:
        folder_dir = os.path.join(dataset_dir, subset)
        folders = next(os.walk(folder_dir))[1]
        for folder in folders:
            if folder == 'real':
                # cls = 1
                folder_path = os.path.join(dataset_dir, subset, folder)
                dataset_path = os.path.join(folder_path, 'image')
                images = next(os.walk(dataset_path))[2]
                for image in images:
                    data_meta.append({'path': folder_path, 'ids': image, 'cls': 1})
            elif folder == 'fake':
                folder_path = os.path.join(dataset_dir, subset, folder)
                dataset_path = os.path.join(folder_path, 'image')
                images = next(os.walk(dataset_path))[2]
                for image in images:
                    data_meta.append({'path': folder_path, 'ids': image, 'cls': 0})
    return data_meta


def reshape(image, size=config.IMAGE_SHAPE):
    h, w = image.shape[:2]
    top_pad = (size - h) // 2
    bottom_pad = size - h - top_pad
    left_pad = (size - w) // 2
    right_pad = size - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    return image


def load_gt(data_meta):
    path = data_meta['path']
    ids = data_meta['ids']
    cls = data_meta['cls']
    image_path = os.path.join(path, 'image', ids)
    mask_path = os.path.join(path, 'mask', ids)
    image = load_arr(image_path)
    mask = load_arr(mask_path)
    mask[mask < 0.9] = 0
    mask[mask >= 0.9] = 1
    return image, mask, cls


def load_arr(path):
    image = skimage.io.imread(path)
    image = reshape(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def data_generator(data_meta, mode, batch_size=config.BATCH_SIZE):
    data_meta = np.copy(data_meta)
    if mode == 'trainS':
        while True:
            # 在每一个epoch开始时打乱list
            np.random.shuffle(data_meta)
            for b in range(batch_size):
                if b == 0:
                    # 初始化
                    images = []
                    masks = []
                image, mask, _ = load_gt(data_meta[b])
                images.append(image)
                masks.append(mask)
            yield ([np.array(images), np.array(masks)], [])
    elif mode == 'trainD':
        while True:
            # 在每一个epoch开始时打乱list
            np.random.shuffle(data_meta)
            for b in range(batch_size):
                if b == 0:
                    # 初始化
                    images = []
                    clss = []
                image, mask, cls = load_gt(data_meta[b])
                images.append(image)
                clss.append(cls)
            yield ([np.array(images), np.array(clss)], [])
    else:
        while True:
            # 在每一个epoch开始时打乱list
            np.random.shuffle(data_meta)
            for b in range(batch_size):
                if b == 0:
                    # 初始化
                    images = []
                    masks = []
                    clss = []
                image, mask, cls = load_gt(data_meta[b])
                images.append(image)
                masks.append(mask)
                clss.append(cls)
            yield ([np.array(images), np.array(masks), np.array(clss)], [])
########################################################################
#
########################################################################


def set_log_dir(model_path=None):
    now = datetime.datetime.now()
    epoch = 0
    if model_path:
        regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]SAN\_" + config.NAME.lower() + "\_(\d{4})\.h5"
        m = re.match(regex, model_path)
        if m:
            now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                    int(m.group(4)), int(m.group(5)))
            epoch = int(m.group(6)) - 1 + 1
            print('Re-starting from epoch %d' % epoch)
    log_dir = os.path.join(config.MODEL_DIR, "{}{:%Y%m%dT%H%M}".format(config.NAME.lower(), now))
    checkpoint_path = os.path.join(log_dir, "SAN_{}_*epoch*.h5".format(config.NAME.lower()))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
    return epoch, log_dir, checkpoint_path


def load_weights(filepath, model, by_name=False, exclude=None):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    import h5py
    # Conditional import to support versions of Keras before 2.2
    # TODO: remove in about 6 months (end of 2018)
    try:
        from keras.engine import saving
    except ImportError:
        # Keras before 2.2 used the 'topology' namespace.
        from keras.engine import topology as saving

    if exclude:
        by_name = True

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    keras_model = model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
        else keras_model.layers

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()


def find_last():
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(config.MODEL_DIR))[1]
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(config.MODEL_DIR))
    # Pick last directory
    dir_name = os.path.join(config.MODEL_DIR, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("SAN"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno
        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name))
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    print("Weights path is {}".format(checkpoint))
    return checkpoint

########################################################################
# LOSS and METRICS
########################################################################


# TODO:自定义评价指标

# Dice部分
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Discrimination部分
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)
        y_true = K.switch(K.greater(smoothing, 0),
                          lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                          lambda: y_true)
    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)


def getPrecision(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision


def getRecall(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall


def getAcc(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return binary_accuracy(y_true, y_pred)


def seg_loss_graph(input_mask, segmentation):
    dice_loss = dice_coef_loss(input_mask, segmentation)
    return dice_loss


def dom_loss_graph(domain_cls, validity):
    y_true = K.flatten(domain_cls)
    y_pred = K.flatten(validity)
    return K.mean(
        K.binary_crossentropy(y_true, y_pred), axis=-1)


def adv_loss_graph(input_mask, domain_cls, segmentation, validity):
    loss = distanceCE(input_mask, domain_cls, segmentation, validity)
    return loss


def Adistance(input_mask, domain_cls, segmentation, validity):
    input_mask = K.flatten(input_mask)
    segmentation = K.flatten(segmentation)
    mask = tf.reshape(input_mask, [config.BATCH_SIZE, -1])
    segment = tf.reshape(segmentation, [config.BATCH_SIZE, -1])
    domain_cls = K.flatten(domain_cls)
    validity = K.flatten(validity)
    x = validity

    intersection = K.sum(mask * segment)
    segLoss = 1 - ((2. * intersection + smooth) / (K.sum(mask) + K.sum(segment) + smooth))
    domLoss = K.maximum(0., 2 * (1 - 2 * K.binary_crossentropy(domain_cls, validity, from_logits=False)))
    loss = (segLoss * (2. * x) / (1. + x)) + (config.ALPHA * domLoss * (1. - x) / (1. + x))
    return K.mean(loss, axis=-1)


def distanceCE(input_mask, domain_cls, segmentation, validity):
    input_mask = K.flatten(input_mask)
    segmentation = K.flatten(segmentation)
    mask = tf.reshape(input_mask, [config.BATCH_SIZE, -1])
    segment = tf.reshape(segmentation, [config.BATCH_SIZE, -1])
    validity = K.flatten(validity)
    x = validity

    intersection = K.sum(mask * segment)
    segLoss = 1-((2. * intersection + smooth) / (K.sum(mask) + K.sum(segment) + smooth))

    y = K.constant(validity) if not K.is_tensor(validity) else validity
    validity = K.maximum(2.0 * y - 1.0, 1.0 - 2.0 * y)
    domain_cls = K.zeros_like(x=validity, dtype=validity.dtype)
    domLoss = K.binary_crossentropy(domain_cls, validity)

    """
    While maintaining the consistency of the form of the formula in the paper, 
    we consider the convergence speed of the training and modify the loss function appropriately, 
    but do not change the actual running result.
    """
    loss = (segLoss * (2. * x) / (1. + x)) + (config.ALPHA * domLoss * (1. - x) / (1. + x))
    return K.mean(loss, axis=-1)

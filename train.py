from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

from tensorflow.keras import regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

# Configuration
BATCH_SIZE = 32
TOP_DROP_OUT = 0.1
NUM_CLASSES = 4
INPUT_SHAPE =  (300, 150, 3)
EFF_INPUT_SHAPE = (224, 224, 3)
LAYER_UNFREEZE = 20
AUGMENTATION = {
    'rot': 0.1, 
    'height': 0.05, 
    'width': 0.05,
    'contrast': 0.1
}


def merge_weights(w, w_init):
    """
    Merge two matching sets of weights

    If shapes match use init
    If shapes differ take the mean of w_init in differing axes
    if shapes still differ broadcast to new shape and add random uniform component scaled by the
    ptp range of weights in w_init
    """
    axes = tuple(i for i, (x, y) in enumerate(zip(w.shape, w_init.shape)) if x != y)
    if axes:
        w_new = np.mean(w_init, axes, keepdims=True)
        if w_new.shape == w.shape:
            w_ptp = np.ptp(w_init, axes, keepdims=True)
            return np.broadcast_to(w_new, w.shape) + (np.random.rand(*w.shape)-0.5)*w_ptp
        return w_new
    return w_init


def build_model(num_classes, augmentation=AUGMENTATION, weights=None, from_logits=True):

    x = layers.Input(shape=INPUT_SHAPE)

    if augmentation is not None:   
        img_augmentation = Sequential(
            [
                layers.RandomRotation(factor=augmentation['rot']),
                layers.RandomTranslation(height_factor=augmentation['height'], width_factor=augmentation['width']),
                layers.RandomFlip(),
                layers.RandomContrast(factor=augmentation['contrast']),
            ],
        name='img_augmentation',
        )

        x = img_augmentation(x)

    model = EfficientNetB0(include_top=False, input_tensor=x, weights=None)

    if weights is None:
        # Transfer weights for imagenet
        model_init = EfficientNetB0(include_top=False, input_shape=EFF_INPUT_SHAPE, weights='imagenet')
        weights = [merge_weights(x, y) for x, y in zip(model.get_weights(), model_init.get_weights())]

    model.set_weights(weights)

    # Freeze the pretrained weights
    model.trainable = False

    x = model.output

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(TOP_DROP_OUT, name='top_dropout')(x)

    fan_in = int(x.shape[-1])
    scale = np.sqrt(1 / fan_in) / 6
    x = layers.Dense(num_classes, name='classifier',
            kernel_initializer=initializers.RandomUniform(minval=0, maxval=scale, seed=None),
            bias_initializer=initializers.Constant(value=-4),
            kernel_constraint=NonNeg(),
            activation=None if from_logits else 'sigmoid',
            activity_regularizer=regularizers.l1(0.001))(x)
    
    # Compile
    model = Model(model.inputs, x, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=from_logits),
        metrics=['accuracy']
    )

    return model

def unfreeze_model(model):
    # We unfreeze the top LAYER_UNFREEZE layers while leaving BatchNorm layers frozen
    for layer in model.layers[-LAYER_UNFREEZE:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

dataset_name = 'bee_dataset'
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    dataset_name, split=('train[:50%]', 'train[50%:75%]', 'train[75%:]'), 
    with_info=True, as_supervised=True, shuffle_files=True
)

# Stack targets
def input_preprocess(image, output):
    label = tf.stack([output['cooling_output'],
                      output['pollen_output'], 
                      output['varroa_output'], 
                      output['wasps_output']])
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)

ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(input_preprocess)
ds_val = ds_val.batch(batch_size=BATCH_SIZE, drop_remainder=True)

model = build_model(num_classes=NUM_CLASSES)

model.summary()

# Train top layer
epochs = 10  
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)


model = unfreeze_model(model)

model.summary()

# Train deep layers
epochs = 10  
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)

# Build model without augmentation layer 
model = Model(model.get_layer('rescaling').input, 
              layers.Activation("sigmoid")(model.output),
              name='EfficientNet')

# Save model
model.training=False
model.save('/output/bee.h5')
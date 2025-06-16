import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

from data_utils import load_data, isolate_and_normalize
from unet_model import build_unet_model
from loss_functions import bce_dice_loss, dice_coefficient

BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
BATCH_SIZE = 12
EPOCHS = 50
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

log_dir = f"./logs/finalN{BASE_FILTERS}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Load and preprocess data
t2_vols, cap_vols = load_data()
t2_vols, cap_vols = t2_vols.transpose(0, 3, 2, 1), cap_vols.transpose(0, 3, 2, 1)
X, y = isolate_and_normalize(t2_vols, cap_vols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED)

# Data augmentation function
def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)

    # Enforce fixed output size (this fixes your error)
    image = tf.image.resize(image, [308, 384])
    mask = tf.image.resize(mask, [308, 384])

    image.set_shape([308, 384, 1])
    mask.set_shape([308, 384, 1])
    return image, mask


# Dataset pipelines
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build model
model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coefficient, 'accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',     # or 'val_dice_coefficient' if you prefer
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save weights
model.save_weights(f"./models/seg_model_N{BASE_FILTERS}.weights.h5")
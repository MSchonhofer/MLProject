import tensorflow as tf
#from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
import numpy as np
import datetime

from data_utils import load_data, isolate_and_normalize
from unet_model import build_unet_model
from loss_functions import combo_loss, dice_coefficient

#mixed_precision.set_global_policy("mixed_float16")

BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
BATCH_SIZE = 8
EPOCHS = 30
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

log_dir = f"./logs/finalN{BASE_FILTERS}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

t2_vols, cap_vols = load_data()
t2_vols, cap_vols = t2_vols.transpose(0, 3, 2, 1), cap_vols.transpose(0, 3, 2, 1)
X, y = isolate_and_normalize(t2_vols, cap_vols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=combo_loss,
    metrics=[dice_coefficient, 'accuracy']
)

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=15,
        restore_best_weights=True
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save_weights(f"./models/seg_model_N{BASE_FILTERS}.weights.h5")
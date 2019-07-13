import os

import tensorflow as tf

import config
from prepare_data import get_datasets

train_generator, val_generator, \
train_num, val_num, class_weights = get_datasets()

# Load the model
model = tf.contrib.saved_model.load_keras_model('./saved_model/flower_photos_serving/1562986584')
# To use evaludate generator you must compile the model first
model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
            metrics=['accuracy'])
# Get the accuracy on the test set
loss, acc = model.evaluate_generator(val_generator,
                                     steps=val_num // config.BATCH_SIZE)

print("The accuracy on test set is: {:6.3f}%".format(acc*100))

import os

import tensorflow as tf

import config
from prepare_data import get_datasets

train_generator, val_generator, \
train_num, valid_num, class_weights = get_datasets()

# Load the model
print(os.path.join('keras_model', config.keras_model_dir))
new_model = tf.keras.models.load_model(os.path.join('keras_model', config.keras_model_dir))
# new_model = tf.contrib.saved_model.load_keras_model(os.path.join('saved_model', config.serving_model_dir))
# new_model.compile()
new_model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001),
            metrics=['accuracy'])
# Get the accuracy on the test set
loss, acc = new_model.evaluate_generator(val_generator,
                                         steps= valid_num // config.BATCH_SIZE)

print("The accuracy on test set is: {:6.3f}%".format(acc*100))

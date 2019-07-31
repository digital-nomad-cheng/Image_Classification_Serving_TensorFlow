import tensorflow as tf

from nets.shufflenetv2 import ShuffleNetV2
from config import NUM_CLASSES, image_height, image_width, channels

def create_model(model_name='mobilenet', use_regularizer=False):
    if model_name == 'mobilenet':
        model = get_mobilenet()
    elif model_name == 'shufflenet':
        model = get_shufflenet()    
    
    if use_regularizer:
        model.save_weights('.tmp.h5')
        regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=1e-5)
        for layer in model.layers:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(layer, attr) and layer.trainable:
                    setattr(layer, attr, regularizer)
        model = tf.keras.models.model_from_json(model.to_json())
        model.load_weights('.tmp.h5', by_name=True)
    
    for layer in model.layers:
        layer.trainable=True

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                 optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001),
                 # optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                 metrics=['accuracy'])
    return model

def get_mobilenet():
    input_tensor = tf.keras.layers.Input(shape=(image_height, image_width, channels))
    model = tf.keras.applications.MobileNetV2(input_shape=(image_height, image_width, channels), input_tensor=input_tensor, include_top=False)
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    dense = tf.keras.layers.Dense(NUM_CLASSES, name='dense', activation=tf.keras.activations.softmax)(avg_pool)
    # logits = tf.keras.layers.Activation('softmax', name='logits')(dense)
    model = tf.keras.models.Model(input_tensor, dense)

    return model

def get_senet():
    pass

def get_densenet():
    pass

def get_shufflenet():
    model = ShuffleNetV2(classes=NUM_CLASSES)
    
    return model

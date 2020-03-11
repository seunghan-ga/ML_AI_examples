import tensorflow as tf
import numpy as np


def W_init(shape=None, name=None):
    """Initialize weights as in paper"""
    # values = np.random.normal(loc=0,scale=1e-2,size=shape)
    values = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=shape)
    return tf.keras.backend.variable(values, name=name)

#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return tf.keras.backend.variable(values, name=name)

def getModel():
    input_shape = (105, 105, 1)
    left_input = tf.keras.layers.Input(input_shape)
    right_input = tf.keras.layers.Input(input_shape)

    #build convnet to use in each siamese 'leg'
    convnet = tf.keras.models.Sequential()
    convnet.add(tf.keras.layers.Conv2D(64, (10, 10), activation='relu',
                                       input_shape=input_shape,
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-2, seed=None),
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(128, (7, 7), activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-2, seed=None),
                                       bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=1e-2, seed=None),))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu',
                                       kernel_initializer=W_init,
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       bias_initializer=b_init))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(256, (4, 4), activation='relu',
                                       kernel_initializer=W_init,
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       bias_initializer=b_init))
    convnet.add(tf.keras.layers.Flatten())
    convnet.add(tf.keras.layers.Dense(4096, activation="sigmoid",
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      kernel_initializer=W_init,
                                      bias_initializer=b_init))

    #encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    #merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: tf.keras.backend.abs(x[0]-x[1])
    both = tf.keras.layers.merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
    siamese_net = tf.keras.models.Model(input=[left_input,right_input],output=prediction)

    # optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
    # optimizer = tf.keras.optimizers.Adam(0.00006)

    # //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    # siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    # siamese_net.count_params()

    return siamese_net


if __name__ == "__main__":
    model = getModel()
    model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(0.00006))
    model.count_params()
    model.summary()

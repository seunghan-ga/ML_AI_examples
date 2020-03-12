import tensorflow as tf


def get_model(input_shape=(105, 105, 1)):
    left_input = tf.keras.layers.Input(input_shape)
    right_input = tf.keras.layers.Input(input_shape)

    convnet = tf.keras.models.Sequential()
    convnet.add(tf.keras.layers.Conv2D(64, (10, 10), activation='relu',
                                       input_shape=input_shape,
                                       kernel_initializer=tf.keras.initializers.random_normal(0., 1e-2),
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(128, (7, 7), activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       kernel_initializer=tf.keras.initializers.random_normal(0., 1e-2),
                                       bias_initializer=tf.keras.initializers.random_normal(0.5, 1e-2)))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(128, (4, 4), activation='relu',
                                       kernel_initializer=tf.keras.initializers.random_normal(0., 1e-2),
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       bias_initializer=tf.keras.initializers.random_normal(0.5, 1e-2)))
    convnet.add(tf.keras.layers.MaxPooling2D())
    convnet.add(tf.keras.layers.Conv2D(256, (4, 4), activation='relu',
                                       kernel_initializer=tf.keras.initializers.random_normal(0., 1e-2),
                                       kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                                       bias_initializer=tf.keras.initializers.random_normal(0.5, 1e-2)))
    convnet.add(tf.keras.layers.Flatten())
    convnet.add(tf.keras.layers.Dense(4096, activation="sigmoid",
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      kernel_initializer=tf.keras.initializers.random_normal(0., 1e-2),
                                      bias_initializer=tf.keras.initializers.random_normal(0.5, 1e-2)))

    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    L1_layer = lambda x: tf.keras.backend.abs(x[0]-x[1])
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = tf.keras.layers.Dense(1, activation='sigmoid',
                                       bias_initializer=tf.keras.initializers.random_normal(0.5, 1e-2))(L1_distance)
    siamese_net = tf.keras.models.Model(inputs=[left_input, right_input], outputs=prediction)

    # optimizer = tf.keras.optimizers.Adam(0.00006)
    # siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

    return siamese_net


if __name__ == "__main__":
    model = get_model()
    model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(0.00006))
    model.count_params()
    model.summary()

import tensorflow as tf


def get_model(shape, filters=6, kernel_size=3):
    left_input = tf.keras.layers.Input(shape)
    right_input = tf.keras.layers.Input(shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(50, activation='relu'))

    input_a = model(left_input)
    input_b = model(right_input)

    distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([input_a, input_b])

    model = tf.keras.models.Model(inputs=[left_input, right_input], outputs=distance)

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=optimizer)

    return model


def euclidean_distance(vects):
    x, y = vects

    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes

    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + (1 - y_true)
                                 * tf.keras.backend.square(tf.keras.backend.maximum(1 - y_pred, 0)))


def train(model, X, y, batch_size=128, epochs=300, verbose=1):
    img_1 = X[:, 0]
    img_2 = X[:, 1]

    model.fit([img_1, img_2], y, validation_split=.25, batch_size=batch_size, verbose=verbose, nb_epoch=epochs)

    return model


if __name__ == "__main__":
    model = get_model(shape=(105, 105, 1))
    model.summary()

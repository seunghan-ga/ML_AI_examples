from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


def euclidean_distance(vects):
    """Calculating Euclidean distance."""
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=1, keepdims=True)

    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    """Output shape of Euclidean distance."""
    shape1, shape2 = shapes

    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """Loss function."""
    margin = 1
    square_pred = tf.keras.backend.square(y_pred)
    margin_square = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))

    return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network_no_conv(shape):
    """Create base network. no Convolution layers."""
    input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return tf.keras.models.Model(input, x)


def create_base_network(shape):
    """Create base network. with Convolution layers."""
    input = tf.keras.layers.Input(shape)

    x = tf.keras.layers.Conv2D(6, (3, 3), activation='relu')(input)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(12, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)

    return tf.keras.models.Model(input, x)


def build_network(shape, conv=True):
    """Build siamese network."""
    input_a = tf.keras.layers.Input(shape)
    input_b = tf.keras.layers.Input(shape)

    base_network = create_base_network(shape) if conv else create_base_network_no_conv(shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = tf.keras.layers.Lambda(euclidean_distance,
                                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    return tf.keras.models.Model(inputs=[input_a, input_b], outputs=distance)


def load_data(path, size=(32, 32), sub_type=False):
    """Load train or test data."""
    x = []
    y = []
    categories = os.listdir(path)

    for category in categories:
        print("loading category: " + category, "class: ", categories.index(category))
        category_path = os.path.join(path, category)
        if sub_type:
            for sub_type in os.listdir(category_path):
                sub_type_path = os.path.join(category_path, sub_type)
                for filename in os.listdir(sub_type_path):
                    image_path = os.path.join(sub_type_path, filename)
                    image = cv2.resize(cv2.imread(image_path), size, cv2.INTER_CUBIC)
                    x.append(image.astype(np.float32) / 255.)
                    y.append(categories.index(category))
        else:
            for filename in os.listdir(category_path):
                image_path = os.path.join(category_path, filename)
                image = cv2.resize(cv2.imread(image_path), size, cv2.INTER_CUBIC)
                x.append(image.astype(np.float32) / 255.)
                y.append(categories.index(category))

    return np.array(x), np.array(y), categories


def create_indices(y, n_classes):
    """Create indices."""
    indices = []

    for i in range(n_classes):
        indices.append(np.where(y == i)[0])

    return np.array(indices)


def create_pairs(x, indices):
    """Create pairs, half same class, half different class."""
    n_classes, n_samples = indices.shape
    pairs = []
    labels = []

    for i in range(n_classes):
        for j in range(n_samples):
            for k in range(j, n_samples):
                query, support = indices[i][j], indices[i][k]
                pairs += [[x[query], x[support]]]
                inc = np.random.randint(1, n_classes, 1)[0]
                in_i = (i + inc) % n_classes
                query, support = indices[i][j], indices[in_i][k]
                pairs += [[x[query], x[support]]]
                labels += [1, 0]

    pairs, labels = shuffle(pairs, labels)

    return np.array(pairs), np.array(labels)


def train(model, x, y, x_v=None, y_v=None, batch_size=128, epochs=20000, verbose=1):
    """Training one-way one-shot learning model."""
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['accuracy'])
    cb_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/{epoch:02d}-{val_loss:.4f}.h5',
                                                       monitor='val_loss',
                                                       verbose=verbose,
                                                       save_best_only=True,
                                                       save_weights_only=True)
    callbacks = [cb_checkpoint, cb_earlystopping]
    if x_v is None and y_v is None:
        model.fit([x[:, 0], x[:, 1]], y, validation_split=.25,
                  batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)
    if x_v is not None and y_v is not None:
        model.fit([x[:, 0], x[:, 1]], y, validation_data=([x_v[:, 0], x_v[:, 1]], y_v),
                  batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    return model


def compute_accuracy(y_true, y_pred):
    """Test one-way one-shot learning accuracy of a siamese neural net."""
    pred = y_pred.ravel() < 0.5

    return np.mean(pred == y_true)


def make_one_way_task(query_image, x, indices, category_idx):
    """Create pairs of query image, support set for testing one-way one-shot learning."""
    n_classes, n_samples = indices.shape
    pairs = []
    image_idx = np.random.randint(0, n_samples, 1)[0]
    support_image = x[indices[category_idx][image_idx]]
    pairs.append([query_image, support_image])

    return np.array(pairs)


def make_n_way_task(query_image, x, indices, categorise):
    """Create pairs of query image, support set for testing n-way one-shot learning."""
    n_classes, n_samples = indices.shape
    pairs = []
    for category in categorise:
        category_idx = categorise.index(category)
        image_idx = np.random.randint(0, n_samples, 1)[0]
        support_image = x[indices[category_idx][image_idx]]
        pairs.append([query_image, support_image])

    return np.array(pairs)


def predict(model, query, x, indices, categories):
    """Prediction. n-way"""
    pairs = make_n_way_task(query, x, indices, categories)
    pred = model.predict([pairs[:, 0], pairs[:, 1]])
    pred = pred.ravel()
    min_distance = int(np.argmin(pred))

    return categories[min_distance], pred[min_distance]


def plot_oneshot_task(pairs):
    """Display query image and support set"""
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.matshow(pairs[0].reshape(105, 105), cmap='gray')
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(pairs[0].reshape(105, 105), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    print("siamese net test")

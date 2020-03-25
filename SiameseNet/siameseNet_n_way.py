from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import cv2
import os


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

    optimizer = tf.keras.optimizers.Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    return siamese_net


def loadimgs(path, n=0):
    """
    Load train or test data
    :param path: path of train directory or test directory
    :param n: current y
    :return: loaded data, labels, category dictionary
    """
    X = []
    y = []
    cat_dict = {}
    curr_y = n

    for category in os.listdir(path):
        print("loading category: " + category)
        cat_dict[category] = [curr_y, None]
        category_path = os.path.join(path, category)

        for pcb_tyep in os.listdir(category_path):
            category_images = []
            pcb_tyep_path = os.path.join(category_path, pcb_tyep)

            for filename in os.listdir(pcb_tyep_path):
                image_path = os.path.join(pcb_tyep_path, filename)
                image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (105, 105), cv2.INTER_LINEAR)
                category_images.append(image.astype(np.float32) / 255.)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            cat_dict[category][1] = curr_y - 1

    y = np.vstack(y)
    X = np.stack(X)

    return X, y, cat_dict


def get_batch(X, n):
    """
    Create batch of n pairs, half same class, half different class
    :param X: train data or test data
    :param n: batch size
    :return: test set and support set pair, targets
    """
    n_classes, n_examples, w, h = X.shape

    categories = np.random.choice(n_classes, size=(n,), replace=False)
    pairs = [np.zeros((n, h, w, 1)) for i in range(2)]
    targets = np.zeros((n,))
    targets[n // 2:] = 1
    for i in range(n):
        category = categories[i]
        idx_1 = np.random.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = np.random.randint(0, n_examples)
        if i >= n // 2:
            category_2 = category
        else:
            category_2 = category if i >= n // 2 else (category + np.random.randint(1, n_classes)) % n_classes
        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def make_oneshot_task(X, N, category=None, cat_dict=None):
    """
    Create pairs of test image, support set for testing N way one-shot learning.
    :param X: train data or test data
    :param N: number of category
    :param category: category name
    :param cat_dict: category dictionary
    :return: test and support set pair, targets
    """
    n_classes, n_examples, w, h = X.shape
    indices = np.random.randint(0, n_examples, size=(N,))

    if category is not None and cat_dict is not None:
        low, high = cat_dict[category]
        if high == low:
            categories = [low]
        else:
            categories = np.random.choice(range(low, high+1), size=(N,), replace=False)
    else:
        categories = np.random.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]

    ex1, ex2 = np.random.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)

    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)

    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)

    pairs = [test_image, support_set]

    return pairs, targets, true_category


def evaluate(model, X, N, k, verbose=True, category=None, cat_dict=None):
    """
    Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
    :param model: trained siamese-net models
    :param X: train data or test data
    :param N: number of category
    :param k: epochs
    :param verbose: verbose (bool, default 0)
    :param category: category name
    :param cat_dict: category dictionary
    :return accuracy
    """
    n_correct = 0
    if verbose is True:
        print("Evaluating models on {} unique {} way one-shot learning tasks ...".format(k, N))

    for i in range(k):
        if category is not None and cat_dict is not None:
            pairs, targets, _ = make_oneshot_task(X, N, category=category, cat_dict=cat_dict)
        else:
            pairs, targets, _ = make_oneshot_task(X, N)
        probs = model.predict(pairs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1

    percent_correct = (100.0 * n_correct / k)

    if verbose is True:
        print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))

    return percent_correct


def train(model, X_train, X_validate=None, epochs=20000, batch_size=6, n=10, k=250, verbose=True, checkpoints=False):
    """
    Training models
    :param model: compiled siamese-net models
    :param X_train: train data set
    :param X_validate: validation data set
    :param batch_size: batch_size
    :param n: number of category
    :param k: validate epoch
    :param epochs: train epoch
    :param verbose: verbose (bool, default: True)
    :param checkpoints: save models weight
    :return: trained models
    """
    if n > 1:
        n = n - 1
    siamese_net = model
    best = 75.
    for i in range(epochs):
        st = time.time()
        pairs, targets = get_batch(X_train, batch_size)
        loss = siamese_net.train_on_batch(pairs, targets)

        if checkpoints is True:
            val_acc = evaluate(siamese_net, X_validate, n, k, verbose=False)
            if val_acc > best:
                best = val_acc
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                siamese_net.save('./checkpoints/weight_{}_{:.4f}_{:.1f}.h5'.format(i, loss, best))
                print("saving")
        ed = time.time()

        if verbose is True:
            if checkpoints is True:
                print("[{:.2f}% | {}/{}]: {:.2f}s, loss: {:.4f}, accuracy: {:.1f}%"
                      .format(100 * ((i + 1) / epochs), i, epochs, ed - st, loss, best))
            else:
                print("[{:.2f}% | {}/{}]: {:.2f}s, loss: {:.4f}"
                      .format(100 * ((i + 1) / epochs), i, epochs, ed - st, loss))

    return siamese_net


def predict(model, test, X_train, cat_dict, n=10):
    """
    Image classification
    :param X: reference data (all categorise)
    :param test: input image
    :param model: trained siamese-net models
    :param cat_dict: category dictionary
    :param n: number of type
    :return: prediction result (category, percent)
    """
    if n > 1:
        n = n - 1
    categorise = list(cat_dict.keys())
    percent = list()
    for category in categorise:
        pairs, _, _ = make_oneshot_task(X_train, n, category, cat_dict)

        support_set = pairs[1]
        test_set = np.zeros(pairs[0].shape)
        for i in range(n):
            test_set[i] = test

        prediction = model.predict([test_set, support_set])
        percent.append(prediction[np.argmax(prediction)] * 100)

    idx = int(np.argmax(percent))

    return categorise[idx], float(percent[idx])


def concat_images(X):
    """
    Concatenates a bunch of images into a big matrix for plotting purposes.
    :param X: test set, and support set
    :return: test image and support set
    """
    nc, h , w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w, n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w, y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1

    return img


def plot_oneshot_task(pairs):
    """
    Display test image and support set
    :param pairs: test set, and support set
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    model = get_model()
    model.count_params()
    model.summary()

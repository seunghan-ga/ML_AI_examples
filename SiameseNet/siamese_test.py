from SiameseNet import siamese
import numpy as np


if __name__ == "__main__":
    x_train, y_train, categories = siamese.load_data("dataset/train")
    x_test, y_test, _ = siamese.load_data("dataset/test")

    indices = siamese.create_indices(np.array(y_train), len(categories))
    pairs, labels = siamese.create_pairs(x_train, indices)
    input_shape = x_train.shape[1:]

    indices_v = siamese.create_indices(np.array(y_test), len(categories))
    pairs_v, labels_v = siamese.create_pairs(x_test, indices_v)

    model = siamese.build_network(shape=input_shape)
    model.summary()

    trainable = False
    if trainable:  # training
        model = siamese.train(model, pairs, labels, batch_size=128, epochs=20000, verbose=2)
        pred = model.predict([pairs_v[:, 0], pairs_v[:, 1]])
        acc = siamese.compute_accuracy(labels_v, pred)
        print("training accuracy:", acc * 100)
    else:  # testing
        model.load_weights("./models/one_way_model.h5")
        n_correct = 0
        sum_incorrect = 0
        epochs = 250
        for i in range(epochs):
            cat_list = [0, 20, 40, 60, 80, 100]
            start = np.random.choice(cat_list, 1)[0]
            query_idx = np.random.randint(start, start + 20, 1)[0]
            query = x_test[query_idx]
            pred = siamese.predict(model, query, x_train, indices, categories)
            # print(start, categories[cat_list.index(start)], query_idx, "->", pred)
            if categories[cat_list.index(start)] == pred[0]:
                n_correct += 1
            else:
                sum_incorrect += pred[1]
        print("correct: {}/{}, accuracy: {:.1f}%"
              .format(n_correct, epochs, (n_correct * 100) / float(epochs)))
        print("average error distance:", sum_incorrect / (epochs - n_correct))

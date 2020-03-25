from SiameseNet import siameseNet_n_way
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_n_way():
    X_train, y_train, cat_train = siameseNet_n_way.loadimgs("./train_2")
    X_test, y_test, cat_test = siameseNet_n_way.loadimgs("./test_2")

    print(X_train.shape, y_train.shape, cat_train)
    print(X_test.shape, y_test.shape, cat_test)

    model = siameseNet_n_way.get_model()
    model = siameseNet_n_way.train(model, X_train, X_test, batch_size=6, n=6, k=250, epochs=25000,
                                   verbose=True, checkpoints=True)

    acc = siameseNet_n_way.evaluate(model, X_test, 6, 250, False)
    print("{:.1f}%".format(acc))


if __name__ == "__main__":
    train_n_way()

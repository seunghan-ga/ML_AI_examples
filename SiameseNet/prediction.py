from SiameseNet import siameseNet_n_way
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prediction_1(model, X_train, X_test, categories, cat_train, cat_test, n=6, k=10):
    all_correct = 0
    for category in categories:
        pairs, _, _ = siameseNet_n_way.make_oneshot_task(X_test, 1, category, cat_test)

        n_correct = 0
        res_cat = []
        percent = []
        st = time.time()
        for i in range(k):
            res = siameseNet_n_way.predict(model, pairs[0], X_train, cat_train, n=n)
            res_cat.append(res[0])
            percent.append(res[1])
            if res[1] > 70. and res[0] == category:
                n_correct += 1

        percent_avg = np.sum(percent) / k
        max_idx = int(np.argmax(percent))
        ed = time.time()

        out = "\n[{}->*{}]: {:.2f}s\npercent_avg: {:.0f}%\n*percent_max: {:.0f}%\ncorrect: {}\nincorrect: {}" \
            .format(category, res_cat[max_idx], ed-st, percent_avg, percent[max_idx], n_correct, k - n_correct)

        print(out)
        all_correct += n_correct

    accu = (all_correct/(k * len(categories))) * 100.
    print("pos: {:.1f}%, neg: {:.1f}%".format(accu, 100 - accu))


def find_cat(index, cat_dict):
    categories = list(cat_dict.keys())
    for cat in categories:
        low, high = cat_dict[cat]
        if low <= index <= high:
            return cat


def prediction_2(model, X_train, X_test, cat_train, cat_test, n=6, epochs=10):
    corr = 0
    incorr = 0
    perncet = []

    for epoch in range(epochs):
        pairs, _, test_cat = siameseNet_n_way.make_oneshot_task(X_test, 1)
        _, w, h, c = pairs[0].shape
        test_set = np.zeros((n, w, h, c))

        for i in range(n):
            test_set[i] = pairs[0]

        pairs, targets, support_cat = siameseNet_n_way.make_oneshot_task(X_train, n)
        support_set = pairs[1]

        prediction = model.predict([test_set, support_set])
        pred_value = (prediction[np.argmax(targets)] * 100.)[0]
        perncet.append(pred_value)

        if np.argmax(prediction) == np.argmax(targets):
            corr += 1
        else:
            incorr += 1

    print("percent_avg: {:1f}%".format(np.sum(perncet) / epochs))
    print("percent_max: {:1f}%".format(np.max(perncet)))
    print("correct:", corr)
    print("incorrect:", incorr)
    print("pos: {:.1f}%, neg: {:.1f}%".format((corr/epochs) * 100., (incorr/epochs) * 100.))


def prediction_n_way():
    model = siameseNet_n_way.get_model()

    X_train, y_train, cat_train = siameseNet_n_way.loadimgs("./train_2")
    X_test, y_test, cat_test = siameseNet_n_way.loadimgs("./test")

    print(X_train.shape, y_train.shape, cat_train)
    print(X_test.shape, y_test.shape, cat_test)

    categories = list(cat_train.keys())
    model.load_weights("./models/weight_25782_0.0457_93.2.h5")

    prediction_1(model, X_train, X_test, categories, cat_train, cat_test, n=20, k=10)
    print("-------------------------------------------------------------------------")
    prediction_2(model, X_train, X_test, cat_train, cat_test, n=6)


if __name__ == "__main__":
    prediction_n_way()

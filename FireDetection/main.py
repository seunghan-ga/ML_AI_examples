# -*- coding: utf-8 -*-
from FireDetection import FireDetection
from tensorflow.python.keras.utils import np_utils
import tensorflow as tf
import numpy as np
import math
import glob
import cv2
import os


def create_dataset(path, label, x=299, y=299, channel=3, nb_classes=2):
    basePath = path[0]
    outPath = path[1]
    readDir = basePath + '*'
    print(readDir)

    dataset = []
    labels = []
    fire = glob.glob(readDir)

    for item in fire:
        print(item, type(cv2.imread(item)), cv2.imread(item).shape)
        tmp = cv2.imread(item)
        tmp.resize(x, y, channel)
        tmp = (tmp.astype(np.float32) / 255.0)
        dataset.append(tmp)
        labels.append([label])

    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels


def slice_video(path, prefix, x=299, y=299, channel=3):
    basePath = path[0]
    savePath = path[1]
    readDir = basePath + '*'

    print(readDir)
    print(savePath)

    fileLise = glob.glob(readDir)
    files = []
    for item in fileLise:
        files.append(item)

    fileno = 0
    dataset = []

    for videoName in files:
        print(videoName)
        keep = True
        video = cv2.VideoCapture(videoName)
        while(keep):
            fileno += 1
            ret, frame = video.read()
            if not ret:
                keep = False
                break
            else:
                small_frame = cv2.resize(frame, (x, y), cv2.INTER_AREA)
                dataset.append(small_frame)
                filename = savePath + "%s_%06d.png" % (prefix, fileno)
                status = cv2.imwrite(filename, small_frame)
                print(status, filename)

    dataset = np.array(dataset)
    print(fileno)
    print(dataset.shape)


def fire_detection(filepath, model, x=299, y=299):
    videoName = filepath
    windowName = "Live Fire Detection"
    keepProcessing = True

    video = cv2.VideoCapture(videoName)
    print("Loaded video ...")

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000 / fps)

    while (keepProcessing):
        start_t = cv2.getTickCount()

        ret, frame = video.read()
        if not ret:
            print("... end of video file reached")
            break

        small_frame = cv2.resize(frame, (x, y), cv2.INTER_AREA)
        input_frame = FireDetection().preprocess_input(np.expand_dims(small_frame, axis=0))
        output = model.predict([input_frame])

        def check(output):
            if round(output[0][0]) == 1: return "flame"
            elif round(output[0][1]) == 1: return "smoke"
            else: return "normal"

        print("detection result : [flame: %.2f / smoke: %.2f / nofire: %.2f] [%s]"
              % ((output[0][0]*100.), (output[0][1]*100.), (output[0][2]*100.), check(output)))

        # cv2.putText(frame, 'FIRE: {0:0.2f}% / nofire: {1:0.2f}%'.format(float(output[0][0]*100.), float(output[0][1]*100.)),
        #             (int(width / 32), int(height / 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 200, 20), 1, cv2.LINE_AA)

        stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000

        cv2.imshow(windowName, frame)


        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF
        if (key == ord('x')):
            keepProcessing = False
        # elif (key == ord('f')):
        #     cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def opencv_detection(video_file):
    video = cv2.VideoCapture(video_file)

    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break

        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)

        output = cv2.bitwise_and(frame, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)
        cv2.imshow("output", output)
        # print("output:", frame)
        # print("output:".format(mask))
        if int(no_red) > 20000:
            print('Fire detected : ', int(no_red))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    x = 299
    y = 299
    channel = 3
    nb_classes = 3
    dataset_flg = False

    # filename = ['data/videos/fire/', 'data/images/fire/']
    # slice_video(filename, 'fire')

    if dataset_flg is True:
        flame_data, flame_labels = create_dataset(['./data/images/train/fire/', ''], 0)
        smoke_data, smoke_labels = create_dataset(['./data/images/train/smoke/', ''], 1)
        normal_data, normal_labels = create_dataset(['./data/images/train/smoke/', ''], 2)

        all_data = np.concatenate((flame_data, smoke_data, normal_data), axis=0)
        all_labels = np.concatenate((flame_labels, smoke_labels, normal_labels), axis=0)
        all_labels = np_utils.to_categorical(all_labels, 3)

        print(all_data.shape)
        print(all_labels.shape)

        np.save('./data/datasets/fire_detection_train_data.npy', all_data)
        np.save('./data/datasets/fire_detection_train_labels.npy', all_labels)

        flame_data, flame_labels = create_dataset(['./data/images/test/fire/', ''], 0)
        smoke_data, smoke_labels = create_dataset(['./data/images/test/smoke/', ''], 1)
        normal_data, normal_labels = create_dataset(['./data/images/test/smoke/', ''], 2)

        all_data = np.concatenate((flame_data, smoke_data, normal_data), axis=0)
        all_labels = np.concatenate((flame_labels, smoke_labels, normal_labels), axis=0)
        all_labels = np_utils.to_categorical(all_labels, 3)

        print(all_data.shape)
        print(all_labels.shape)

        np.save('./data/datasets/fire_detection_test_data.npy', all_data)
        np.save('./data/datasets/fire_detection_test_labels.npy', all_labels)

    model = FireDetection().inception_resnet_v2(include_top=True, weights=None, input_shape=(x, y, channel), classes=nb_classes)
    print("Constructed Model ...")

    model.load_weights('models/test/080-0.67043.hdf5')
    print("Loaded CNN network weights ...")

    fire_detection('data/videos/nofire_400240.mp4', model, x=x, y=y)
    # opencv_detection('data/videos/case2_house.mp4')




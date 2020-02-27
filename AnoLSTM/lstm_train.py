# -*- coding: utf-8 -*-
from AnoLSTM.analysis.AnormalyDetection import LSTM_autoencoder
from AnoLSTM.utils.parser import MYSQL_Parser
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import pool
import tensorflow as tf
import pandas as pd
import numpy as np
import pymysql
import logging
import joblib
import sys


# database connection pool
global mysql_pools

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    try:
        db_info = {
            "host": "192.168.1.101",
            "port": 3306,
            "user": "hadoop",
            "password": "hadoop",
            "db": "DEMO",
            "charset": "utf8"
        }
        mysql_pools = pool.QueuePool(lambda: pymysql.connect(**db_info), pool_size=3, max_overflow=0)
        logging.info("create DB connection pool")
        logging.info("connected DB info. [%s:%s/%s]" % (db_info['host'], db_info['port'], db_info['db']))

    except Exception as e:
        logging.exception('failed - create db connection pool')
        sys.exit()

    try:
        conn = mysql_pools.connect()
        cursor = conn.cursor()

        query = "desc gyro"
        cursor.execute(query)
        res = cursor.fetchall()
        columns = [col[0] for col in res]

        sql = "select * from gyro"
        cursor.execute(sql)
        raw_data = cursor.fetchall()
        conn.close()

    except Exception as e:
        logging.exception('failed - get data.')
        sys.exit()

    df = MYSQL_Parser().parser_none_sequential(raw_data, columns)
    df_train = df[columns[1:]].astype("float").values
    print(df_train)

    # set parameters
    timesteps = 30  # window size
    features = columns[1:]
    n_features = len(features)
    units = len(features) * 2
    epochs = 200
    batch_size = 32

    # data preparation
    train = LSTM_autoencoder().data_prepare(X=df_train, timesteps=timesteps, n_features=n_features)
    flat_train = LSTM_autoencoder().flatten(train)

    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(LSTM_autoencoder().flatten(train))
    scaled_train = LSTM_autoencoder().scale(train, scaler)

    # get model
    LSTM_Model = LSTM_autoencoder().get_model(timesteps=timesteps,
                                              n_features=n_features,
                                              units=units,
                                              activate_func=tf.keras.activations.relu,
                                              recurrent_func=tf.keras.activations.sigmoid,
                                              kernel_init=tf.keras.initializers.glorot_uniform(),
                                              loss_func=tf.keras.losses.mean_squared_error,
                                              optimize_func=tf.keras.optimizers.Adam(lr=0.0001),
                                              dropout=0.3)
    LSTM_Model.summary()

    # fit model
    MODEL_SAVE_PATH = "./models/checkpoints/"
    model_path = MODEL_SAVE_PATH + '{epoch:03d}-{val_loss:.5f}.hdf5'
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1,
                                                       save_best_only=True)
    cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = LSTM_Model.fit(x=scaled_train,
                             y=scaled_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=2,
                             callbacks=[cb_checkpoint, cb_early_stopping],
                             validation_split=0.2,
                             shuffle=True).history

    # save scaler
    joblib.dump(scaler, "models/normalization.scaler")
    print("Saved scaler to disk")

    # model save
    model_json = LSTM_Model.to_json()  # serialize model to JSON
    json_file = open("models/model.json", "w")
    json_file.write(model_json)
    LSTM_Model.save_weights("models/model.h5")  # serialize weights to HDF5
    print("Saved model to disk")

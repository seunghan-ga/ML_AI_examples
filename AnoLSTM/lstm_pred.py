# -*- coding: utf-8 -*-
from AnoLSTM.utils.parser import MYSQL_Parser
from AnoLSTM.analysis.AnormalyDetection import LSTM_autoencoder
from sklearn.preprocessing import MinMaxScaler
from confluent_kafka import KafkaException
from confluent_kafka import TopicPartition
from confluent_kafka import Consumer
from sklearn.metrics import r2_score
from sqlalchemy import pool
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import pymysql
import logging
import datetime
import time
import argparse
import configparser
import sys

# database connection pool
global mysql_pools


# kafka receiver
def receive_data(consumer, windowsize=30, topic=None, partition=None, offset=None):
    try:
        rt_values = []
        close_time = time.time() + windowsize

        if partition is None and offset is None:
            pass
        else:
            consumer.seek(TopicPartition(topic, partition, offset))

        while True:
            msg = consumer.poll(timeout=1.0)

            if time.time() > close_time:
                offset = msg.offset()
                partition = msg.partition()
                break

            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            else:
                rt_values.append(msg.value().decode('utf-8').split(','))

        return rt_values, partition, offset

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')
        return None


# outlier detection func.
def outlier_detection(data, params):
    # scaling
    scaler = joblib.load(str(params['scaler_path']))
    print('Loaded scaler from disk')

    # model load
    json_file = open(str(params['model_json_path']), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(str(params['model_path']))
    LSTM_Model = loaded_model
    print('Loaded model from disk')

    # dataset
    columns = params['features'].split(',')
    df_test = data[columns].astype('float').values

    # parameter
    timesteps = params['timesteps']
    n_features = len(columns)

    # adjust window size
    zeros = [[0 for i in range(n_features)] for j in range(timesteps + 1)]
    df_test = np.concatenate((zeros, df_test), axis=0)

    # 3d transformation
    valid = LSTM_autoencoder().data_prepare(X=df_test, timesteps=timesteps, n_features=n_features)

    # scaling
    scaled_valid = LSTM_autoencoder().scale(valid, scaler)

    # prediction
    pred_valid = LSTM_Model.predict(scaled_valid, verbose=0)

    # data flatten
    flatten_scaled_valid = LSTM_autoencoder().flatten(scaled_valid)
    flatten_pred_valid = LSTM_autoencoder().flatten(pred_valid)

    # scoring
    mse = np.mean(np.power(flatten_scaled_valid - flatten_pred_valid, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    err_values = error_df.Reconstruction_error.values

    # calculate threshold
    r2 = np.abs(r2_score(flatten_scaled_valid, flatten_pred_valid))

    k = 1.5
    if r2 >= 0.5:
        k = 0.556
    elif 0.5 > r2 >= 0.3:
        k = 1.5
    elif 0.3 > r2 >= 0.1:
        k = 1.668
    else:
        k = 2.224

    q3, q1 = np.percentile(err_values, [75, 25])
    iqr = q3 - q1
    threshold = (q3 + (iqr * k)) * 2.67

    print('r2:', r2)
    print('k:', k)
    print('q3:', q3)
    print('threshold:', threshold)

    # outlier detection
    pred_label = []
    for i in err_values:
        if threshold > i > 0:
            pred_label.append(0)
        else:
            pred_label.append(1)

    # contribution variable
    contribution_variable = LSTM_autoencoder().contribution(flatten_scaled_valid, flatten_pred_valid, pred_label,columns)
    # tensorflow-keras session close
    tf.keras.backend.clear_session()

    return error_df.values.tolist(), contribution_variable.values.tolist(), pred_label, threshold


if __name__ == "__main__":
    # set default configuration
    config = configparser.ConfigParser()
    config.read("config/config.cfg")
    hostname = config["DATABASE_INFO"]["hostname"]
    port = config["DATABASE_INFO"]["port"]
    username = config["DATABASE_INFO"]["username"]
    password = config["DATABASE_INFO"]["password"]
    db_name = config["DATABASE_INFO"]["database"]
    tb_name = config["DATABASE_INFO"]["table"]
    interval = int(config["LSTM_INFO"]["interval"])
    window = int(config["LSTM_INFO"]["window"])
    infoname = config["LSTM_INFO"]["ref_info"]
    stime = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
    params = None

    # create database connection pool
    try:
        db_info = {
            "host": str(hostname),
            "user": str(username),
            "password": str(password),
            "db": str(db_name),
            "charset": "utf8"
        }
        mysql_pools = pool.QueuePool(lambda: pymysql.connect(**db_info), pool_size=3, max_overflow=0)

    except Exception as e:
        sys.exit()

    # get reference info.
    try:
        conn = mysql_pools.connect()
        cursor = conn.cursor()

        # get table info.
        query = "desc %s" % (tb_name)
        cursor.execute(query)
        res = cursor.fetchall()
        columns = [col[0] for col in res]

        # get reference info.
        sql = "select * from STD_LSTM_INFO where INFO_NAME='%s'" % (infoname)
        cursor.execute(sql)
        info_data = cursor.fetchall()
        conn.close()

        if len(info_data) != 0:
            info = info_data[0]
            params = {
                'timesteps': int(info[5]),
                'features': str(info[6]),
                'batch': int(info[7]),
                'epochs': int(info[8]),
                'model_path': str(info[9]),
                'model_json_path': str(info[10]),
                'scaler_path': str(info[11])
            }
        if params['timesteps'] is not None:
            window = params['timesteps']

    except Exception as e:
        sys.exit()

    # main loop
    print('--------------------lstm outlier detection start--------------------')
    conf = {'bootstrap.servers': "192.168.1.101:9092", 'group.id': "gyro_01", 'session.timeout.ms': 6000,
            'auto.offset.reset': 'earliest'}
    topic = 'gyro_01'
    windowsize = 10
    partition = None
    offset = None
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    try:
        while True:
            res = receive_data(consumer, windowsize, topic, partition, offset)
            data = res[0]
            partition = res[1]
            offset = res[2]

            if len(data) >= 10:  # outlier detection
                print("read line :", len(data), *("\n%s" % str(line) for line in data))

                now = datetime.datetime.now()
                now_str = datetime.datetime.strftime(now, "%Y-%m-%d %H:%M:%S")
                formated = datetime.datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S")

                df = MYSQL_Parser().parser_none_sequential(data, columns)
                score, con_val, label, thr = outlier_detection(df, params)
                print(score, con_val, label, thr)

                # result insert
                insert_data = []
                for i in range(len(data)):
                    # ctime, val1, val2, val3, score, thr, err, contribute, jobtime
                    t = [data[i][0], float(data[i][1]), float(data[i][2]), float(data[i][3]),
                         float(score[i][0]), float(thr), int(label[i]), str(con_val[i]), formated]
                    insert_data.append(t)
                print("insert line :", len(insert_data), *("\n%s" % line for line in insert_data))

                conn = mysql_pools.connect()
                cursor = conn.cursor()
                sql = "insert into STS_LSTM_OUTLIER_RES values(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.executemany(sql, insert_data)
                conn.commit()
                conn.close()

            else:
                print("no data")

    except Exception as e:
        sys.stderr.write('[ERROR]: %s\n' % e)

    finally:
        consumer.close()

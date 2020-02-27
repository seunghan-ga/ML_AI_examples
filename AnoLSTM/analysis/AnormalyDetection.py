# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd


class LSTM_autoencoder(object):
    def temporalize(self, X, y, lookback):
        '''
        Temporalize the data
        :param X:
        :param y:
        :param lookback:
        :return: Temporalized data
        '''

        output_X = []
        output_y = []
        for i in range(len(X) - lookback - 1):
            t = []
            for j in range(1, lookback + 1):  # Gather past records upto the lookback period
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + lookback + 1])

        return output_X, output_y

    def flatten(self, X):
        '''
        Flatten a 3D array.
        :param X: A 3D array for lstm, where the array is sample x timesteps x features.
        :return: A 2D array, sample x features.
        '''
        # sample x features array.
        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]

        return (flattened_X)

    def scale(self, X, scaler):
        '''
        Scale 3D array.
        :param X: A 3D array for lstm, where the array is sample x timesteps x features.
        :param scaler: A scaler object, e.g., StandardScaler, normalize ...
        :return: Scaled 3D array.
        '''
        for i in range(X.shape[0]):
            X[i, :, :] = scaler.transform(X[i, :, :])

        return X

    def data_prepare(self, X=None, y=None, timesteps=1, n_features=1):
        '''
        convert data (2d > 3d)
        :param X:
        :param y:
        :param timesteps: Window size
        :param n_features: Column length
        :return:
        '''
        # if len(X) < timesteps:

        if y is None:
            input_y = np.zeros(len(X))
        else:
            input_y = y
        X, y = self.temporalize(X=X, y=input_y, lookback=timesteps)
        X = np.array(X)
        X_3D = X.reshape((X.shape[0], timesteps, n_features))

        return X_3D

    def get_model(self, timesteps=1, n_features=1, **kwargs):
        '''
        Get LSTM Autoencoder
        :param units: Dimensionality of the output space.
        :param timesteps: Window size
        :param n_features: Column length
        :param **kwargs: Parameters
        :return: LSTM model
        '''
        # set parameter
        units = kwargs['units']
        activate_fn = kwargs['activate_func']  # default: tanh
        recurrent_fn = kwargs['recurrent_func']
        optimize_fn = kwargs['optimize_func']
        loss_fn = kwargs['loss_func']
        kernel_init = kwargs['kernel_init']  # default: glorot_uniform
        dropout = kwargs['dropout']

        LSTM_Model = tf.keras.Sequential()
        # encoder
        LSTM_Model.add(tf.keras.layers.LSTM(units, activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout,
                                            input_shape=(timesteps, n_features)))
        LSTM_Model.add(tf.keras.layers.LSTM(int(units / 2), activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=False, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.RepeatVector(timesteps))
        # decoder
        LSTM_Model.add(tf.keras.layers.LSTM(int(units / 2), activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.LSTM(units, activation=activate_fn, recurrent_activation=recurrent_fn,
                                            kernel_initializer=kernel_init, return_sequences=True, dropout=dropout))
        LSTM_Model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
        LSTM_Model.compile(optimizer=optimize_fn, loss=loss_fn)

        return LSTM_Model

    def contribution(self, true, pred, pred_label, feature):
        '''
        Contribution of variable
        :param true: true data
        :param pred: predict data
        :param pred_label_test: predict data label
        :param feature: cloumns
        :return: contribution
        '''
        # df_pred_label = pd.DataFrame(pred_label)
        df_true = pd.DataFrame(true)
        df_pred = pd.DataFrame(pred)

        # point = np.where(df_pred_label == 1)[0]
        mse = list()
        for i in range(len(pred_label)):
            if pred_label[i] == 1:
                mse.append(np.array(np.power(df_true.loc[i] - df_pred.loc[i], 2)))
            else:
                mse.append(np.array([0 for i in range(len(feature))]))

        c_vals = pd.DataFrame(mse)
        c_vals.columns = feature

        return c_vals
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime


class MYSQL_Parser(object):
    def parser_sequential(self, data, columns):
        res_data = []
        res_columns = ["ID"]
        for col in columns:
            res_columns.append(col)

        for idx in range(len(data)):
            tmp = []
            tmp.append(idx)
            for item in data[idx]:
                if type(item) == datetime.datetime: tmp.append(str(item))
                else: tmp.append(item)
            res_data.append(tmp)

        res = np.array(res_data)
        res_df = pd.DataFrame(data=res, columns=res_columns)

        return res_df

    def parser_none_sequential(self, data, columns):
        res_data = []
        for idx in range(len(data)):
            tmp = []
            for item in data[idx]:
                if type(item) == datetime.datetime: tmp.append(str(item))
                else: tmp.append(item)
            res_data.append(tmp)

        res = np.array(res_data)
        res_df = pd.DataFrame(data=res, columns=columns)

        return res_df

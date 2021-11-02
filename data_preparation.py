import pandas as pd
import csv
import datetime as dt
from datetime import timedelta
from os.path import exists

from pandas.core.frame import DataFrame

start_date = dt.date(2020, 4, 15)
days_to_check = 100


# Takes the path to the data and predict value csv files and merges them to generate a DataFrame that can be fed into ML models.
def merge_data_and_predict_value(data_filepath: str, predict_value_filepath: str) -> DataFrame:
    data = pd.read_csv(data_filepath)
    predict_values = pd.read_csv(predict_value_filepath)
    merged = pd.merge(data, predict_values, on='date')
    return merged


def combine_multiple_csv(start_date, end_date, filepath):
    date_1 = start_date
    filepaths = []
    while date_1 <= end_date:
        date_2 = date_1+timedelta(days=1)
        while date_2 <= end_date:
            path = filepath + str(date_1) + '-' + str(date_2) + '.csv'
            if exists(path):
                filepaths.append(path)
                break
            date_2 += timedelta(days=1)
        date_1 = date_2 + timedelta(days=1)
    print(filepaths)
    data_frames = []
    for file_path in filepaths:
        data_frames.append(pd.read_csv(file_path))
    merged_data = pd.concat(data_frames)
    merged_data.reset_index(drop=True, inplace=True)
    merged_data.to_csv('daily_sentiment_'+str(start_date) +
                       '-'+str(end_date)+'.csv', index=False)



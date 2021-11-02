from typing import Dict, List
import pandas as pd
from pandas.core.frame import DataFrame
from data_preparation import merge_data_and_predict_value
import datetime as dt
import numpy as np
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing


def load_data(data: str, predict_values: str) -> DataFrame:
    merge_data = merge_data_and_predict_value(
        data + data_path_end, predict_values + predict_path_end)
    merge_data.drop(['date'], axis=1, inplace=True)
    nmpy_data = merge_data.to_numpy()
    return nmpy_data


def optimize(model, X_train, X_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    space = dict()
    if not isinstance(model, LinearRegression):
        space['alpha'] = loguniform(1e-5, 100)
        if isinstance(model, Ridge):
            space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
    space['fit_intercept'] = [True, False]
    search = RandomizedSearchCV(model,
                                space,
                                scoring='neg_mean_absolute_error',
                                n_jobs=-1,
                                cv=cv)
    search.fit(X_train, y_train)
    return search, X_train, X_test, y_train, y_test


def get_results(trained_model, X_train, X_test, y_train, y_test):
    train_score = r2_score(y_train, trained_model.predict(X_train))
    test_score = r2_score(y_test, trained_model.predict(X_test))
    train_mae = mean_absolute_error(y_train, trained_model.predict(X_train))
    test_mae = mean_absolute_error(y_test, trained_model.predict(X_test))
    train_rsme = mean_squared_error(y_train, trained_model.predict(X_train))
    test_rsme = mean_squared_error(y_test, trained_model.predict(X_test))
    return train_score, test_score, train_mae, train_rsme, test_mae, test_rsme


def export_plot(filename, trained_model, X, y):
    plt.plot(trained_model.predict(X), label='prediction')
    plt.plot(y, label='actual data')
    plt.xlabel('Day')
    plt.ylabel(used_predict_values)
    plt.savefig(filename)


def export_results(filename, results):
    return 0


data = [
    '/measures_data/discussed_measures_', '/sentiment_data/daily_sentiment_'
]

predict_values = [
    '/prediction_data/new_cases_',
    '/prediction_data/new_cases_derivative_',
    '/prediction_data/new_cases_derivative_7_days',
    '/prediction_data/new_cases_derivative_trailing_moving_mean',
    '/prediction_data/new_cases_derivative_trailing_moving_mean_7_days',
]

start_date = dt.date(2020, 7, 29)
end_date = dt.date(2021, 1, 25)

data_path_end = str(start_date) + '-' + str(end_date) + '.csv'
predict_path_end = str(start_date) + '-' + str(end_date) + '_US.csv'

used_data = data[0]
used_predict_values = predict_values[0]

data = load_data(used_data, used_predict_values)
X = data[:, :-1]
if np.shape(X)[1] == 2:
    X = X.reshape(-1, 1)
print(X)
y = data[:, -1]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=1)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lin_model, X_train, X_test, y_train, y_test = optimize(LinearRegression(),
                                                       X_train, X_test,
                                                       y_train, y_test)
lin_results = get_results(lin_model, X_train, X_test, y_train, y_test)
print('Lin_reg:' + str(lin_results))

ridge_model, X_train, X_test, y_train, y_test = optimize(
    Ridge(), X_train, X_test, y_train, y_test)
ridge_results = get_results(ridge_model, X_train, X_test, y_train, y_test)
print('Ridge:' + str(ridge_results))

lasso_model, X_train, X_test, y_train, y_test = optimize(
    Lasso(), X_train, X_test, y_train, y_test)
lasso_results = get_results(lasso_model, X_train, X_test, y_train, y_test)
print('Lasso' + str(lasso_results))

#lasso_results = evaluate(Lasso(), X, y)
#ridge_results = evaluate(Ridge(), X, y)
#svr_results = evaluate(SVR(), X, y)

# filename = used_data+'_'+used_predict_values + \
#    '_'+str(start_date)+'-'+str(end_date)+'_results'
#export_results (filename,[linear_results,lasso_results,ridge_results,svr_results])

# Do evaluation on data by using it with available models by following steps
# 1. Optimize  model with hyperparameter-optimization to get best hyperparameters
# 2. Create model with optimized parameters and train the model with the given data
# 3. Do K-Crossfold-validation
# 4. Calculate different scores (R2, MAE, MRSE, etc)
# 5. Save scores in csv file
# 6. Make and save interesting plot
# 7. Re-do 1-6 for other models.

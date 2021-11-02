from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

filename = 'data_derivative.csv'
data = pd.read_csv(filename)
features = [
    "lockdown", "mask", "quarantine", "close", "shutdown", "distancing",
    "gatherings"
]

X = data[features]
y = data['derivative_cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

reg = make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2))
reg.fit(X_train, y_train)

print("train score" + str(reg.score(X_train, y_train)))
print("test score" + str(reg.score(X_test, y_test)))
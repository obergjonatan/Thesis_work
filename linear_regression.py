import numpy as np
from pandas.core.frame import DataFrame
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=1)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]

reg = LinearRegression()
ridge = Ridge()
lasso = Lasso()

search = RandomizedSearchCV(ridge,
                            space,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            cv=cv)

result = search.fit(X_train, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

reg = reg.fit(X_train, y_train)
ridge = Ridge(alpha=9.1, fit_intercept=True,
              solver='sag').fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

print("Linear regression train score" + str(reg.score(X_train, y_train)))
print("Linear regression test score" + str(reg.score(X_test, y_test)))

print("Ridge regression train score" + str(ridge.score(X_train, y_train)))
print("Ridge regression test score" + str(ridge.score(X_test, y_test)))

print("Lasso regression train score" + str(lasso.score(X_train, y_train)))
print("Lasso regression test score" + str(lasso.score(X_test, y_test)))

##plt.plot(reg.predict(X), label='prediction')
##plt.plot(Y, label='true')
##plt.axis([0, 100, 0, 0.02])
# plt.show()

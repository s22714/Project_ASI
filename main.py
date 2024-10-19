import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("OnlineNewsPopularity.csv", index_col=False).drop('url', axis=1)
df.columns = df.columns.str.strip()

train, test = train_test_split(df, test_size=0.3)

features = list(df.columns[:-1])

X_train = train.loc[:, features]
y_train = train['shares']

X_test = test.loc[:, features]
y_test= test['shares']

model = LinearRegression().fit(X_train,y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('RMSE = ', rmse)


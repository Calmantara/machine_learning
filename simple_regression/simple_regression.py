
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv("./simple_regression/Salary_Data.csv")
# df_data = raw_data.copy()

# copy data
df_data = np.copy(raw_data)

# get features and output
x_data = df_data[:, :1]
y_data = df_data[:, -1:]

# split data from data set
length_data = len(x_data)
x_train = x_data[:int(length_data*0.7)]
x_test = x_data[int(length_data*0.7):]
y_train = y_data[:int(length_data*0.7)]
y_test = y_data[int(length_data*0.7):]

regressor = LinearRegression()
regressor.fit(X=x_train, y=y_train)
y_predic = regressor.predict(x_test)
print(y_predic, y_test)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_predic, color='blue')
plt.show()

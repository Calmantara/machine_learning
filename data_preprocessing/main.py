#
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

raw_data = pd.read_csv("Data.csv")
# df_data = raw_data.copy()

df_data = np.copy(raw_data)
x_data = df_data[:, :-1]
y_data = df_data[:, -1:]

# raw_data.Salary.plot()
# plt.show()
# print(df_data)

# x_data = df_data.iloc[:, :-1]
# x_data['Salary'].fillna(method='ffill', inplace = True)
# print(x_data)
# y_data = df_data.iloc[:, -1:]
# print(y_data)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x_data[:, 1:3])
x_data[:, 1:3] = imputer.transform(x_data[:, 1:3])
# print(x_data)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

x_data = np.array(ct.fit_transform(x_data))
# print(x_data)

le = LabelEncoder()
y_data = le.fit_transform(y_data)
# print(y_data)

std = StandardScaler()
x_data = np.array(std.fit_transform(x_data))
# print(x_data)

data_length = len(x_data)
x_train = x_data[:int(data_length*0.8), :]
x_test = x_data[int(data_length*0.8):, :]

y_data_length = len(y_data)
y_train = y_data[:int(y_data_length * 0.8)]
y_test = y_data[int(y_data_length * 0.8):]

print("y data ", y_data)
print("y train", y_train)
print("y test", y_test)

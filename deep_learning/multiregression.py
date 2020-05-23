import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("./deep_learning/Churn_Modelling.csv")
x_data = dataset.iloc[:, 3:13].values
y_data = dataset.iloc[:, 13:].values

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1, 2])], remainder='passthrough')
x_data = ct.fit_transform(x_data)
x_data = x_data[:, 1:]
x_data = np.delete(x_data, 3, 1)

data_length = len(x_data)
x_train = x_data[:int(0.75*data_length), :]
x_test = x_data[int(0.75*data_length):, :]
y_train = y_data[:int(0.75*data_length), :]
y_test = y_data[int(0.75*data_length):, :]

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

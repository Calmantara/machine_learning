from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

model = Sequential()
model.add(Dense(32, input_shape=(11,),
                activation='relu', kernel_initializer='uniform'))
model.add(Dense(16,
                activation='relu', kernel_initializer='uniform'))
model.add(Dense(1,
                activation='sigmoid', kernel_initializer='uniform'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=512, epochs=100)

y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

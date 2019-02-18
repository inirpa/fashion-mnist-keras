# https://medium.com/nanonets/how-to-classify-fashion-images-easily-using-convnets-81e1e0019ffe  
# https://github.com/khanhnamle1994/fashion-mnist/blob/master/CNN-1Conv.ipynb
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
data_train = pd.read_csv('data/fashion_train.csv')
data_test = pd.read_csv('data/fashion_test.csv')

X = np.array(data_train.iloc[:,1:])
y = to_categorical(np.array(data_train.iloc[:,0]))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

X_test = np.array(data_test.iloc[:,1:])
y_test = to_categorical(np.array(data_test.iloc[:,0]))

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_test = X_test.astype('float32')
X_test /= 255

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
X_val = X_val.astype('float32')
X_val /= 255

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.2))

cnn1.add(Flatten())

cnn1.add(Dense(units=128, activation='relu'))
cnn1.add(Dense(units=10, activation='softmax'))

cnn1.summary()
cnn1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history1 = cnn1.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1, validation_data=(X_val, y_val))
score1 = cnn1.evaluate(X_test, y_test, verbose=0)
print('Test loss : ', score1[0])
print('Test accuracy : ', score1[1])
import keras
from keras import Input
from keras.src.initializers import HeNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def work(optimizer, epoch, X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(64, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(1, kernel_initializer=HeNormal()))
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=optimizer)
    model.fit(X_train, Y_train, epochs=epoch, batch_size=128, verbose=0)
    return model.evaluate(X_test, Y_test, batch_size=128, verbose=0)

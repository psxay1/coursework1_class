from keras import regularizers
import data_loader as dl
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = dl.load_data('data/bank-full.csv')  # import data file
df.head()
df0 = dl.into_dataframe(df)
pd.set_option('display.max_columns', None)

df0_features = dl.get_features(df0, 'y')

df0_label = dl.get_labels(df0, 'y')


def category(m, column):
    for col in column:
        if m[col].dtype == np.dtype('object'):
            d = pd.get_dummies(m[col], prefix=col)
            d.head()
            m = pd.concat([m, d], axis=1)
            # drop the encoded column
            m.drop([col], axis=1, inplace=True)
    return m


x = df0_features

y = df0_label

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.DataFrame(y))
y_new = encoder.transform(pd.DataFrame(y)).toarray()

scaler = MinMaxScaler()
x = category(x, df0_features.head(0))
x[['balance', 'duration', 'pdays', 'previous']] = scaler.fit_transform(x[['balance', 'duration', 'pdays', 'previous']])

x = x.to_numpy()

kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_new[train_index], y_new[test_index]
    model = tf.keras.models.Sequential()
    model.add(Dense(51, kernel_regularizer=regularizers.l2(0.1), input_dim=51, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(180, kernel_regularizer=regularizers.l2(0.1), activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(50, kernel_regularizer=regularizers.l2(0.1), activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=30, batch_size=250, validation_split=0.45)

    # Generate generalization metrics
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss train and validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.plot(np.arange(0, len(history.history['loss'])), history.history['loss'])
plt.title("Loss")
plt.show()
plt.plot(np.arange(0, len(history.history['acc'])), history.history['acc'])
plt.title("Accuracy")
plt.show()

import tensorflow as tf
import Classification_kFold_Final as pp
import numpy as np
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train = pp.x_train
y_train = pp.y_train
x_test = pp.x_test
y_test = pp.y_test
z = pp.s

kFold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
f1scores = []
iteration = 1

for train_index, test_index in kFold.split(x_train, y_train):
    model = tf.keras.models.Sequential()
    model.add(Dense(51, input_dim=51, activation='relu'))
    model.add(Dense(270, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    train_x, test_x = x_train[train_index], x_test[test_index]
    train_y, test_y = y_train[train_index], y_test[test_index]
    history = model.fit(train_x, train_y, epochs=45, batch_size=140, validation_split=0.2)
    score = model.evaluate(test_x, test_y, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}*100')


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

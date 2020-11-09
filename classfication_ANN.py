import tensorflow as tf
import preprocessing as pp
import numpy as np
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train = pp.x_train
y_train = pp.y_train
x_test = pp.x_test
y_test = pp.y_test
z = pp.s

# Function to create model


model = tf.keras.models.Sequential()
model.add(Dense(51, input_dim=51, activation='relu'))
model.add(Dense(270, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=45, batch_size=140, validation_split=0.2)

# Generate generalization metrics
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Graph training: cost train and validation

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

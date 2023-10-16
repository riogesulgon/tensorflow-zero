import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# print the tensorflow version
print(f'tensorflow: {tf.__version__}')
print(f'python: {sys.version}')

# create features
x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# turn numpy arrays into tensors
X = tf.constant(x)
Y = tf.constant(y)

# get the shapes of the tensors
print("Shapes of tensors:")
print(X.shape, Y.shape)

input_shape = X[0].shape
output_shape = Y[0].shape
print("Input and output shapes:")
print(input_shape, output_shape)

tf.random.set_seed(42)

# create a model using the sequential API
# 1 dense layer with 1 neuron
# A dense layer is also known as a fully connected layer or
# a hidden layer
model = tf.keras.Sequential([tf.keras.layers.Dense(1)]) # keras is a high level API for tensorflow
# model.add(tf.keras.layers.Dense(1)) # add another layer

# compile the model
# MAE - mean absolute error, The average of the absolute differences between predictions and actual values.
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(), # SGD - stochastic gradient descent,
              # SGD is the same as mini-batch gradient descent, but the batch size is set to 1
              metrics=['mae'])

# fit the model
# expand_dims adds an extra dimension to the tensor
expanded = tf.expand_dims(X, axis=-1)
# print expanded
print(expanded)
model.fit(expanded, Y, epochs=100) # training the model for 100 epochs

# print a line separator
print('-' * 120)

# check out X and Y
print(X)
print(Y)

# make a prediction
y_pred = model.predict([17.0])

# print the prediction
print(f'Prediction for 17.0: {y_pred}')

# plot the model
# plt.scatter(X, Y)
# plt.show()
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

plt.scatter(x, y, label='Training data', color='blue', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()

# Visualize it. Open a window using matplotlib and display the scatter plot
plt.show()

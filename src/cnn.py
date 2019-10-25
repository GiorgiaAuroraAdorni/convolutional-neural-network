#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

# Load the CIFAR-10 dataset
(x_train_valid, y_train_valid), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split the original training set into a validation set (1000 images) and an effective training set (49000 images).
# The data should be shuffled: each observation-class pair in the original training set should be assigned at random
# to one of these sets.

n_train = 49000

indices = np.random.permutation(x_train_valid.shape[0])

train_indices = indices[:n_train]
validation_indices = indices[n_train:]

x_valid = x_train_valid[validation_indices, :, :]
y_valid = y_train_valid[validation_indices]
x_train = x_train_valid[train_indices, :, :]
y_train = y_train_valid[train_indices]

# The observations contain pixel values between 0 and 255. Scale these values to lie between 0 and 1.
scaled_x_valid = np.divide(x_valid, 255)
scaled_x_train = np.divide(x_train, 255)

# The class assignments are represented by integers between 0 and 9. Create binary assignment matrices (as required
# by softmax cross entropy with logits v2).
labels = np.arange(10)

y_valid_matrix = (y_valid == labels).astype(int)
y_train_matrix = (y_train == labels).astype(int)
y_test_matrix = (y_test == labels).astype(int)

#  Using TensorFlow, define the following architecture for a convolutional neural network (using rectified linear
#  activation functions):
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

# (a) Convolutional layer 1: 32 filters, 3 × 3.
W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.zeros(shape=(32,)))
A_conv1 = tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# (b) Convolutional layer 2: 32 filters, 3 × 3.
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
b_conv2 = tf.Variable(tf.zeros(shape=(32,)))
A_conv2 = tf.nn.relu(tf.nn.conv2d(A_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# (c) Max-pooling layer 1: 2 × 2 windows.
A_pool1 = tf.nn.max_pool(A_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# (d) Convolutional layer 3: 64 filters, 3 × 3.
W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv3 = tf.Variable(tf.zeros(shape=(64,)))
A_conv3 = tf.nn.relu(tf.nn.conv2d(A_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

# (e) Convolutional layer 4: 64 filters, 3 × 3.
W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
b_conv4 = tf.Variable(tf.zeros(shape=(64,)))
A_conv4 = tf.nn.relu(tf.nn.conv2d(A_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

# (f) Max-pooling layer 2: 2 × 2 windows.
A_pool2 = tf.nn.max_pool(A_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
A_pool2_flat = tf.reshape(A_pool2, [-1, 8 * 8 * 64]) # ? x 3136

# (g) Fully connected layer 1: 512 units.
W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros(shape=(512,)))
A_fc1 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc1) + b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.zeros(shape=(10,)))
Z = tf.matmul(A_fc1, W_fc2) + b_fc2

# (h) Softmax output layer.
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
loss = tf.reduce_mean(loss)

hits = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

# In order to train the network, you will need to generate batches based on your training set.
# Each epoch should split the training dataset into batches differently.
# This is easily accomplished by creating a new list of (randomly generated) batches for each epoch.

# Use the following hyperparameters to train your network:
learning_rate = 1e-3
batch_size = 32
epochs = 50

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(0, epochs):
    print('Epoch: {}.'.format(epoch))
    permutation = np.random.permutation(n_train)

    for i, sample_index in enumerate(range(0, n_train, batch_size)):
        batch_indices = permutation[sample_index:sample_index + batch_size]
        batch = [scaled_x_train[batch_indices], y_train_matrix[batch_indices]]

        train_loss, _ = session.run([loss, train], feed_dict={X: batch[0], Y: batch[1]})

        if i % 50 == 0:
            print('Iteration {}. Train Loss: {:.2f}.'.format(i, train_loss))

    validation_loss = session.run(loss, feed_dict={X: scaled_x_valid, Y: y_valid_matrix})
    print('Validation loss: {}.'.format(validation_loss))

    # classification accuracy on the validation set
    validation_accuracy = session.run(accuracy, feed_dict={X: scaled_x_valid, Y: y_valid_matrix})
    print('Validation accuracy: {:.2f}%.'.format(validation_accuracy * 100))

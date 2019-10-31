#!/usr/bin/env python3
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def check_dir(out_dir, model_dir):
    """

    :param out_dir: project directory for the output files
    :param model_dir: model directory
    :return:
    """
    dir = out_dir + model_dir
    set_dir = [dir + 'train.txt',
               dir + 'validation.txt',
               dir + 'test.txt']

    if not os.path.exists(dir):
        os.makedirs(dir)

    return set_dir


def plot_setting(valid_accuracies, out_dir, model_dir, epoch, final=False):
    """

    :param valid_accuracies: OrderedDict containing the model and the corrispondent validation accuracy
    :param out_dir: project directory for the output files
    :param model_dir: model directory
    :param epoch: number of epochs of the model
    :param final: boolean parameter that is True if only if the model is the one selected for the final test
    """
    dir = out_dir + model_dir

    set_dir = [dir + 'train.txt',
               dir + 'validation.txt',
               dir + 'test.txt']

    img_dir = out_dir + 'img/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_dir = img_dir + model_dir.strip('/') + '-'

    my_plot(valid_accuracies, model_dir, set_dir, img_dir, 'Loss', epoch, final)
    my_plot(valid_accuracies, model_dir, set_dir, img_dir, 'Accuracy', epoch, final)


def my_plot(valid_accuracies, model_dir, set_dir, img_dir, title, epoch, final):
    """

    :param valid_accuracies: OrderedDict containing the model and the corrispondent validation accuracy
    :param model_dir: model directory
    :param set_dir: a list containing the path for the 3 output files containing the performances
    :param img_dir: directory for the output images
    :param title: title of the plot
    :param epoch: number of epochs of the model
    :param final: boolean parameter that is True if only if the model is the one selected for the final test
    """
    with open(set_dir[0], 'r') as f:
        train_lines = f.readlines()[1:]

    with open(set_dir[1], 'r') as f:
        valid_lines = f.readlines()[1:]

    train_loss = np.array([])
    train_accuracy = np.array([])

    valid_loss = np.array([])
    valid_accuracy = np.array([])

    for line in train_lines:
        el = line.strip('\n').split(',')
        train_loss = np.append(train_loss, float(el[1]))
        train_accuracy = np.append(train_accuracy, float(el[2]))

    for line in valid_lines:
        el = line.strip('\n').split(',')
        valid_loss = np.append(valid_loss, float(el[1]))
        valid_accuracy = np.append(valid_accuracy, float(el[2]))

    x = np.arange(1, epoch + 1, dtype=int)

    if final:
        with open(set_dir[2], 'r') as f:
            test_lines = f.readlines()[1:]

        test_loss = np.array([])
        test_accuracy = np.array([])

        for line in test_lines:
            el = line.strip('\n').split(',')
            test_loss = np.append(test_loss, float(el[1]))
            test_accuracy = np.append(test_accuracy, float(el[2]))

    plt.xlabel('epoch/iteration', fontsize=11)

    if title == 'Accuracy':
        valid_accuracies[set_dir[1]] = valid_accuracy[-1]
        print(set_dir[1], valid_accuracy[-1])

        plt.plot(x, train_accuracy, label='Train ' + title)
        plt.plot(x, valid_accuracy, label='Valid ' + title)
        if final:
            plt.plot(x, test_accuracy, label='Test ' + title)
            print(set_dir[2], ' accuracy: ', test_accuracy[-1])
        plt.ylabel('accuracy', fontsize=11)
        plt.ylim(0, 1.01)

    elif title == 'Loss':
        plt.plot(x, train_loss, label='Train ' + title)
        plt.plot(x, valid_loss, label='Valid ' + title)
        if final:
            plt.plot(x, test_loss, label='Test ' + title)
            print(set_dir[2], ' loss: ', test_loss[-1])

        plt.ylabel('loss', fontsize=11)
        # plt.ylim(0, 7)

    plt.legend()
    plt.title(title + ' model ' + model_dir.strip('/'), weight='bold', fontsize=12)
    plt.savefig(img_dir + title + '.png')
    plt.show()


def data_preparation(n_train):
    """

    :param n_train: number of effective sample in the training set
    :return scaled_x_train, scaled_x_valid, y_train_matrix, y_valid_matrix, scaled_x_test, y_test_matrix: the inputs
    and labels for the three datasets
    """
    # Load the CIFAR-10 dataset
    (x_train_valid, y_train_valid), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Split the original training set into a validation set (1000 images) and an effective training set (49000 images).
    # The data should be shuffled: each observation-class pair in the original training set should be assigned at random
    # to one of these sets.
    random_state = np.random.RandomState(seed=1)
    indices = random_state.permutation(x_train_valid.shape[0])

    train_indices = indices[:n_train]
    validation_indices = indices[n_train:]

    x_valid = x_train_valid[validation_indices, :, :]
    y_valid = y_train_valid[validation_indices]
    x_train = x_train_valid[train_indices, :, :]
    y_train = y_train_valid[train_indices]

    # The observations contain pixel values between 0 and 255. Scale these values to lie between 0 and 1.
    scaled_x_valid = np.divide(x_valid, 255)
    scaled_x_train = np.divide(x_train, 255)
    scaled_x_test = np.divide(x_test, 255)

    # The class assignments are represented by integers between 0 and 9. Create binary assignment matrices (as required
    # by softmax cross entropy with logits v2).
    labels = np.arange(10)

    y_valid_matrix = (y_valid == labels).astype(int)
    y_train_matrix = (y_train == labels).astype(int)
    y_test_matrix = (y_test == labels).astype(int)

    return scaled_x_train, scaled_x_valid, y_train_matrix, y_valid_matrix, scaled_x_test, y_test_matrix


def net_param(model, learning_rate):
    """

    :param model:
    :param learning_rate:
    :return:
    """
    with tf.variable_scope("model_{}".format(model)):
        dropout = tf.placeholder(tf.float32, [], name='dropout')  # Placeholder that represent the probability to keep

        # each neuron
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')

        Z = conv_net(X, dropout)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
        loss = tf.reduce_mean(loss)

        s_loss = tf.summary.scalar('loss', loss)

        hits = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

        s_accuracy = tf.summary.scalar('accuracy', accuracy)

        # Merges all summaries into single a operation
        summaries = tf.summary.merge([s_loss])

        # Optimiser
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    return summaries, X, Y, Z, dropout, n_train, loss, accuracy, train


def conv_net(X, dropout):
    """

    :param X:
    :param dropout:
    :return:
    """
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

    A_pool1 = tf.nn.dropout(A_pool1, rate=dropout)

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
    A_pool2 = tf.nn.dropout(A_pool2, rate=dropout)
    A_pool2_flat = tf.reshape(A_pool2, [-1, 8 * 8 * 64])  # ? x 3136

    # (g) Fully connected layer 1: 512 units.
    W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev=0.1))
    b_fc1 = tf.Variable(tf.zeros(shape=(512,)))
    A_fc1 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc1) + b_fc1)

    A_fc1 = tf.nn.dropout(A_fc1, rate=dropout)

    # (h) Softmax output layer.
    W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.zeros(shape=(10,)))
    Z = tf.matmul(A_fc1, W_fc2) + b_fc2

    Z = tf.nn.dropout(Z, rate=dropout)

    return Z


def main(set_dir, learning_rate, batch_size, epochs, drop_values=None, final=False):
    """

    :param set_dir:
    :param learning_rate:
    :param batch_size:
    :param epochs:
    :param drop_values:
    :param final:
    """
    # Network Parameters
    if drop_values is None:
        drop_values = [0, 0]

    summaries, X, Y, Z, dropout, n_train, loss, accuracy, train = net_param(1, learning_rate)

    f_train = open(set_dir[0], "w")
    f_valid = open(set_dir[1], "w")
    f_test = open(set_dir[2], "w")

    f_train.write('epoch, loss, accuracy\n')
    f_valid.write('epoch, loss, accuracy\n')
    f_test.write('epoch, loss, accuracy\n')

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print('Model: {}.'.format(set_dir))

    for epoch in range(0, epochs):
        print('Epoch: {}.'.format(epoch))
        permutation = np.random.permutation(n_train)

        # In order to train the network, you will need to generate batches based on your training set.
        # Each epoch should split the training dataset into batches differently.
        # This is easily accomplished by creating a new list of (randomly generated) batches for each epoch.
        avg_loss = 0
        avg_accuracy = 0

        for i, sample_index in enumerate(range(0, n_train, batch_size)):
            batch_indices = permutation[sample_index:sample_index + batch_size]
            batch = [x_train[batch_indices], y_train[batch_indices]]

            # Train
            train_loss, train_accuracy, _ = session.run([loss, accuracy, train],
                                                        feed_dict={X: batch[0], Y: batch[1], dropout: drop_values[0]})

            avg_accuracy += train_accuracy * y_train[batch_indices].shape[0]
            avg_loss += train_loss * y_train[batch_indices].shape[0]

        train_loss = avg_loss / n_train
        train_accuracy = avg_accuracy / n_train

        print('Train Loss: {:.2f}. Train Accuracy: {:.2f}%.'.format(train_loss, train_accuracy * 100))
        f_train.write(str(epoch) + ', ' + str(train_loss) + ', ' + str(train_accuracy) + '\n')

        # Validation
        validation_loss, validation_accuracy = session.run([loss, accuracy],
                                                           feed_dict={X: x_valid, Y: y_valid, dropout: drop_values[1]})
        print('Validation loss: {}.'.format(validation_loss))

        print('Validation accuracy: {:.2f}%.'.format(validation_accuracy * 100))
        f_valid.write(str(epoch) + ', ' + str(validation_loss) + ', ' + str(validation_accuracy) + '\n')

        # Test
        if final:
            test_loss, test_accuracy = session.run([loss, accuracy],
                                                   feed_dict={X: x_test, Y: y_test, dropout: drop_values[1]})
            print('Test loss: {}.'.format(test_loss))
            print('Test accuracy: {:.2f}%.'.format(test_accuracy * 100))
            f_test.write(str(epoch) + ', ' + str(test_loss) + ', ' + str(test_accuracy) + '\n')

    f_train.close()
    f_valid.close()
    f_test.close()


########################################################################################################################

n_train = 49000
x_train, x_valid, y_train, y_valid, x_test, y_test = data_preparation(n_train)

out_dir = 'USI/DeepLearning/Assignment2/out/'
plot_dir = 'out/'
valid_accuracies = OrderedDict()

### Experiment 1 ###
main(check_dir(out_dir, '1/'), 1e-3, 32, 50)
# plot_setting(valid_accuracies, plot_dir, '1/', 50)

### Experiment 2 ###
main(check_dir(out_dir, '2/'), 1e-3, 32, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '2/', 50)

### Experiment 3 ###
main(check_dir(out_dir, '3/'), 1e-4, 32, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '3/', 50)

### Experiment 4 ###
# main(check_dir(out_dir, '4/'), 1e-4, 64, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '4/', 50)

### Experiment 5 ###
# main(check_dir(out_dir, '5/'), 1e-4, 64, 50, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '5/', 50)

### Experiment 6 ###
# main(check_dir(out_dir, '6/'), 1e-4, 64, 20, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '6/', 20)

### Experiment 7 ###
# main(check_dir(out_dir, '7/'), 1e-2, 64, 20, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '7/', 20)

### Experiment 8 ###
# main(check_dir(out_dir, '8/'), 1e-2, 32, 50, [0.5, 0])

### Experiment 9 ###
# main(check_dir(out_dir, '9/'), 1e-3, 64, 50, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '9/', 50)

### Experiment 10 ###
# main(check_dir(out_dir, '10/'), 1e-3, 32, 50, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '10/', 50)

### Experiment 11 ###
# main(check_dir(out_dir, '11/'), 1e-3, 32, 20, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '11/', 20)

### Experiment 12 ###
# main(check_dir(out_dir, '12/'), 1e-3, 32, 20, [0.3, 0])
# plot_setting(valid_accuracies, plot_dir, '12/', 20)

### Experiment 13 ###
# main(check_dir(out_dir, '13/'), 1e-3, 64, 20, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '13/', 20)

### Experiment 14 ###
# main(check_dir(out_dir, '14/'), 1e-3, 64, 20, [0.3, 0])
# plot_setting(valid_accuracies, plot_dir, '14/', 20)

### Experiment 15 ###
# main(check_dir(out_dir, '15/'), 1e-4, 32, 50, [0.4, 0])
# plot_setting(valid_accuracies, plot_dir, '15/', 50)

### Experiment 16 ###
# main(check_dir(out_dir, '16/'), 1e-5, 32, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '16/', 50)

### Experiment 17 ###
main(check_dir(out_dir, '17/'), 1e-3, 128, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '17/', 50)

### Experiment 18 ###
main(check_dir(out_dir, '18/'), 1e-3, 128, 50, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '18/', 50)

### Experiment 19 ###
# main(check_dir(out_dir, '19/'), 1e-3, 128, 50, [0.7, 0])
# plot_setting(valid_accuracies, plot_dir, '19/', 50)

### Experiment 20 ###
# main(check_dir(out_dir, '20/'), 1e-3, 64, 50, [0.7, 0])
# plot_setting(valid_accuracies, plot_dir, '20/', 50)

### Experiment 21 ###
# main(check_dir(out_dir, '21/'), 1e-3, 64, 50, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '21/', 50)

### Experiment 22 ###
# main(check_dir(out_dir, '22/'), 1e-4, 64, 50, [0.7, 0])
# plot_setting(valid_accuracies, plot_dir, '22/', 50)

### Experiment 23 ###
# main(check_dir(out_dir, '23/'), 1e-4, 64, 50, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '23/', 50)

### Experiment 24 ###
# main(check_dir(out_dir, '24/'), 1e-4, 128, 50, [0.7, 0])
# plot_setting(valid_accuracies, plot_dir, '24/', 50)

### Experiment 25 ###
# main(check_dir(out_dir, '25/'), 1e-4, 128, 50, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '25/', 50)

### Experiment 26 ###
# main(check_dir(out_dir, '26/'), 1e-4, 128, 50, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '26/', 50)

### Experiment 27 ###
# main(check_dir(out_dir, '27/'), 1e-3, 128, 30, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '27/', 50)

### Experiment 28 ###
# main(check_dir(out_dir, '28/'), 0.005, 128, 50, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '28/', 50)

### Experiment 30 ###
# main(check_dir(out_dir, '30/'), 1e-3, 128, 100, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '30/', 100)

### Experiment 31 ###
# main(check_dir(out_dir, '31/'), 1e-4, 128, 100, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '31/', 100)

### Experiment 32 ###
main(check_dir(out_dir, '32/'), 1e-4, 128, 100, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '32/', 100)

### Experiment 33 ###
# main(check_dir(out_dir, '33/'), 1e-4, 128, 200, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '33/', 200)

### Experiment 34 ###
# main(check_dir(out_dir, '34/'), 1e-4, 256, 200, [0.6, 0])
# plot_setting(valid_accuracies, plot_dir, '34/', 200)

### Experiment 35 ###
main(check_dir(out_dir, '35/'), 1e-4, 256, 200, [0.5, 0])
# plot_setting(valid_accuracies, plot_dir, '35/', 200)

### Experiment 36 ###
main(check_dir(out_dir, '36/'), 1e-4, 128, 200, [0.5, 0], True)
# plot_setting(valid_accuracies, plot_dir, '36/', 200, True)

valid_accuracies = sorted(valid_accuracies.items(), key=lambda t: t[1], reverse=True)
print(valid_accuracies[:10])

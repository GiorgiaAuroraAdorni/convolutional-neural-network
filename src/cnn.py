#!/usr/bin/env python3
import os
from collections import OrderedDict
import time
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

    plt.xlabel('epoch', fontsize=11)

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
        plt.ylim(0, 4.5)

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


def net_param(model, learning_rate, neurons_fc, batch_norm):
    """

    :param model: current model
    :param learning_rate: learning rate of the model
    :param neurons_fc: number of units for the fully connected layer
    :param batch_norm: boolean parameter that if is True is computed the batch normalisation
    :return X, Y, Z, dropout_mpool, dropout_fc, loss, accuracy, train
    """
    with tf.variable_scope("model_{}".format(model)):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')
        # dropout = tf.placeholder(tf.float32, [], name='dropout')
        dropout_mpool = tf.placeholder(tf.float32, [], name='dropout_mpool')
        dropout_fc = tf.placeholder(tf.float32, [], name='dropout_fc')

        Z = conv_net(X, neurons_fc, dropout_mpool, dropout_fc, batch_norm)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z)
        loss = tf.reduce_mean(loss)

        hits = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

        # Optimiser
        beta1 = 2e-4
        beta2 = 2e-3
        epsilon = 1e-6
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        train = optimizer.minimize(loss)

        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # train = opt.compute_gradients(loss)

    return X, Y, Z, dropout_mpool, dropout_fc, n_train, loss, accuracy, train


def create_conv_layer(filter_size, input_size, output_size, input, batch_norm):
    """

    :param filter_size: size of the filter
    :param input_size: input size
    :param output_size: number of neurons
    :param input: input of the layer
    :param batch_norm: boolean parameter that if is True is computed the batch normalisation
    :return A_conv, W_conv
    """
    W_conv = tf.Variable(tf.truncated_normal([filter_size, filter_size, input_size, output_size], stddev=0.1))
    b_conv = tf.Variable(tf.zeros(shape=(output_size,)))
    A_conv = tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv

    if batch_norm:
        num_out_nodes = W_conv.shape[-1]
        gamma = tf.Variable(tf.ones([num_out_nodes]))
        beta = tf.Variable(tf.zeros([num_out_nodes]))
        batch_mean, batch_variance = tf.nn.moments(A_conv, [0])
        A_conv = tf.nn.batch_normalization(A_conv, batch_mean, batch_variance, beta, gamma, 1e-3)

    A_conv = tf.nn.relu(A_conv)
    return A_conv, W_conv


def create_fc_layer(input_size, output_size, input, batch_norm):
    """

    :param input_size: input size
    :param output_size: number of neurons
    :param input: input of the layer
    :param batch_norm: boolean parameter that if is True is computed the batch normalisation
    :return A_fc
    """
    W_fc = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b_fc = tf.Variable(tf.zeros(shape=(output_size,)))
    A_fc = tf.matmul(input, W_fc) + b_fc

    if batch_norm:
        num_out_nodes = W_fc.shape[-1]
        gamma = tf.Variable(tf.ones([num_out_nodes]))
        beta = tf.Variable(tf.zeros([num_out_nodes]))
        batch_mean, batch_variance = tf.nn.moments(A_fc, [0])
        A_fc = tf.nn.batch_normalization(A_fc, batch_mean, batch_variance, beta, gamma, 1e-3)

    A_fc = tf.nn.relu(A_fc)
    return A_fc


def conv_net(X, neurons_fc, dropout_mpool, dropout_fc, batch_norm):
    """

    :param X: input
    :param neurons_fc: number of neurons for the fully connected layer
    :param d_mpool: placeholder that represent the probability to keep each neuron after max pooling layers.
    :param d_fc: placeholder that represent the probability to keep each neuron after fully connected layers.
    :param batch_norm: boolean parameter that if is True is computed the batch normalisation
    :return Z: output
    """
    # (a) Convolutional layer 1: 32 filters, 3 × 3.
    A_conv1, W_conv1 = create_conv_layer(3, 3, 32, X, batch_norm)

    # (b) Convolutional layer 2: 32 filters, 3 × 3.
    A_conv2, W_conv2 = create_conv_layer(3, 32, 32, A_conv1, batch_norm)

    # (c) Max-pooling layer 1: 2 × 2 windows.
    A_pool1 = tf.nn.max_pool(A_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    A_pool1 = tf.nn.dropout(A_pool1, rate=dropout_mpool)

    # (d) Convolutional layer 3: 64 filters, 3 × 3.
    A_conv3, W_conv3 = create_conv_layer(3, 32, 64, A_pool1, batch_norm)

    # (e) Convolutional layer 4: 64 filters, 3 × 3.
    A_conv4, W_conv4 = create_conv_layer(3, 64, 64, A_conv3, batch_norm)

    # (f) Max-pooling layer 2: 2 × 2 windows.
    A_pool2 = tf.nn.max_pool(A_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    A_pool2 = tf.nn.dropout(A_pool2, rate=dropout_mpool)
    A_pool2_flat = tf.reshape(A_pool2, [-1, 8 * 8 * 64])

    # (g) Fully connected layer 1: 512 units.
    A_fc1 = create_fc_layer(8 * 8 * 64, neurons_fc, A_pool2_flat, batch_norm)
    A_fc1 = tf.nn.dropout(A_fc1, rate=dropout_fc)

    # (h) Softmax output layer.
    W_fc2 = tf.Variable(tf.truncated_normal([neurons_fc, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.zeros(shape=(10,)))
    Z = tf.matmul(A_fc1, W_fc2) + b_fc2

    return Z


def main(set_dir, learning_rate, batch_size, epochs, d_mpool=0.0, d_fc=0.0, neurons_fc=512, final=False,
         batch_norm=False):
    """

    :param set_dir: a list containing the path for the 3 output files containing the performances
    :param learning_rate: learning rate of the model
    :param batch_size: samples contained in each batch
    :param epochs: number of epochs for the model
    :param neurons_fc: number of neurons for fully connected layer
    :param d_mpool: dropout represent the probability to keep each neuron during the training. The default value is 0 for
                    both, that corresponds to keep the neuron with probability 1 after max pooling layers.
    :param d_fc: dropout represent the probability to keep each neuron during the training. The default value is 0 for
                both, that corresponds to keep the neuron with probability 1 after fully connected layers.
    :param final: boolean parameter that is True if only if the model is the one selected for the final test
    :param batch_norm: boolean parameter that if is True is computed the batch normalisation
    """
    # Network Parameters
    X, Y, Z, dropout_mpool, dropout_fc, n_train, loss, accuracy, train = net_param(1,
                                                                                   learning_rate,
                                                                                   neurons_fc,
                                                                                   batch_norm)

    f_train = open(set_dir[0], "w")
    f_valid = open(set_dir[1], "w")
    f_test = open(set_dir[2], "w")

    f_train.write('epoch, loss, accuracy, time\n')
    f_valid.write('epoch, loss, accuracy, time\n')
    f_test.write('epoch, loss, accuracy, time\n')

    # Avoid allocating all GPU memory upfront.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
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

        print("Starting train…")
        train_start = time.time()
        for i, sample_index in enumerate(range(0, n_train, batch_size)):
            batch_indices = permutation[sample_index:sample_index + batch_size]
            batch = [x_train[batch_indices], y_train[batch_indices]]

            # Train
            train_loss, train_accuracy, _ = session.run([loss, accuracy, train],
                                                        feed_dict={X: batch[0],
                                                                   Y: batch[1],
                                                                   dropout_mpool: d_mpool, dropout_fc: d_fc})

            avg_accuracy += train_accuracy * y_train[batch_indices].shape[0]
            avg_loss += train_loss * y_train[batch_indices].shape[0]

        train_loss = avg_loss / n_train
        train_accuracy = avg_accuracy / n_train
        train_end = time.time()
        train_time = train_end - train_start

        print('Train Loss: {:.2f}. Train Accuracy: {:.2f}%. Train Time: {} sec.'
              .format(train_loss, train_accuracy * 100, train_time))
        f_train.write(str(epoch) + ', ' + str(train_loss) + ', ' + str(train_accuracy) + ',' + str(train_time) + '\n')

        # Validation
        print("Starting validation…")
        valid_start = time.time()
        validation_loss, validation_accuracy = session.run([loss, accuracy],
                                                           feed_dict={X: x_valid,
                                                                      Y: y_valid,
                                                                      dropout_mpool: 0, dropout_fc: 0})
        valid_end = time.time()
        valid_time = valid_end - valid_start

        print('Validation loss: {}. Validation accuracy: {:.2f}%. Validation Time: {} sec.'
              .format(validation_loss, validation_accuracy * 100, valid_time))
        f_valid.write(str(epoch) + ', ' + str(validation_loss) + ', ' + str(validation_accuracy) + ',' + str(valid_time) + '\n')

        # Test
        if final:
            print("Starting test…")
            test_start = time.time()
            test_loss, test_accuracy = session.run([loss, accuracy],
                                                   feed_dict={X: x_test,
                                                              Y: y_test,
                                                              dropout_mpool: 0, dropout_fc: 0})
            test_end = time.time()
            test_time = test_end - test_start

            print('Test loss: {}. Test accuracy: {:.2f}%. Test Time: {} sec.'
                  .format(test_loss, test_accuracy * 100, test_time))
            f_test.write(str(epoch) + ', ' + str(test_loss) + ', ' + str(test_accuracy) + ',' + str(test_time) + '\n')

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
# main(check_dir(out_dir, '1/'), 1e-3, 32, 50)
plot_setting(valid_accuracies, plot_dir, '1/', 50)

### Experiment 2 ###
# main(check_dir(out_dir, '2/'), 1e-3, 32, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '2/', 50)

### Experiment 2b ###
# main(check_dir(out_dir, '2b/'), 1e-3, 32, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '2b/', 300)

### Experiment 3 ###
# main(check_dir(out_dir, '3/'), 1e-4, 32, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '3/', 50)

### Experiment 3b ###
# main(check_dir(out_dir, '3b/'), 1e-4, 32, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '3b/', 300)

### Experiment 3c - batch normalisation ###
# main(check_dir(out_dir, '3c/'), 1e-4, 32, 300, 0.5, 0.5, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '3c/', 300)

### Experiment 4 ###
# main(check_dir(out_dir, '4/'), 1e-3, 128, 50, 0.6, 0.6)
plot_setting(valid_accuracies, plot_dir, '4/', 50)

### Experiment 4b ###
# main(check_dir(out_dir, '4b/'), 1e-3, 128, 300, 0.6, 0,6)
plot_setting(valid_accuracies, plot_dir, '4b/', 300)

### Experiment 5 ###
# main(check_dir(out_dir, '5/'), 1e-3, 128, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '5/', 50)

### Experiment 5b ###
# main(check_dir(out_dir, '5b/'), 1e-3, 128, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '5b/', 300)

### Experiment 5c ###
# main(check_dir(out_dir, '5c/'), 1e-3, 128, 300, 0.5, 0.5, final=True, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '5c/', 300)
plot_setting(valid_accuracies, plot_dir, '5c-test/', 300, True)

### Experiment 5d - different neurons for fc1 512 -> 1024 - no batch norm ###
# main(check_dir(out_dir, '10/'), 1e-3, 128, 50, 0.5, 0.5, 1024)
plot_setting(valid_accuracies, plot_dir, '10/', 50)

### Experiment 5g - different neurons for fc1 512 -> 1024 ###
# main(check_dir(out_dir, '5g/'), 1e-3, 128, 50, 0.5, 0.5, 1024, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '5g/', 50)

### Experiment 5h - different neurons for fc1 512 -> 1024 - mutiepochs ###
# main(check_dir(out_dir, '10c/'), 1e-3, 128, 300, 0.5, 0.5, 1024, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '10c/', 300)

### Experiment 5i - different dropout & different neurons ###
# main(check_dir(out_dir, '5i/'), 1e-3, 128, 50, 0.25, 0.5, 1024, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '5i/', 50)

### Experiment 6 ###
# main(check_dir(out_dir, '6/'), 1e-4, 128, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '6/', 50)

### Experiment 6b ###
# main(check_dir(out_dir, '6b/'), 1e-4, 128, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '6b/', 300)

### Experiment 6c - batch normalisation ######
# main(check_dir(out_dir, '6c/'), 1e-4, 128, 300, 0.5, 0.5, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '6c/', 300)

### Experiment 7 ###
# main(check_dir(out_dir, '7/'), 1e-4, 256, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '7/', 50)

### Experiment 7b ###
# main(check_dir(out_dir, '7b/'), 1e-4, 256, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '7b/', 300)

### Experiment 8 ###
# main(check_dir(out_dir, '8/'), 1e-3, 256, 50, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '8/', 50)

### Experiment 8b ###
# main(check_dir(out_dir, '8b/'), 1e-3, 256, 300, 0.5, 0.5)
plot_setting(valid_accuracies, plot_dir, '8b/', 300)

### Experiment 8c ###
# main(check_dir(out_dir, '8c/'), 1e-3, 256, 300, 0.5, 0.5, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '8c/', 300)

### Experiment 9 - different dropout  ###
# main(check_dir(out_dir, '9/'), 1e-3, 128, 50, 0.25, 0.5)
plot_setting(valid_accuracies, plot_dir, '9/', 50)

# main(check_dir(out_dir, '9b/'), 1e-3, 128, 300, 0.25, 0.5)
plot_setting(valid_accuracies, plot_dir, '9b/', 300)

# main(check_dir(out_dir, '9c/'), 1e-3, 128, 50, 0.25, 0.5, batch_norm=True)
plot_setting(valid_accuracies, plot_dir, '9c/', 50)

################################################################################


# main(check_dir(out_dir, '10/'), 1e-4, 64, 50, 0.5, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '10/', 50)

# main(check_dir(out_dir, '11/'), 1e-4, 64, 50, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '11/', 50)

# main(check_dir(out_dir, '12/'), 1e-4, 64, 20, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '12/', 20)

# main(check_dir(out_dir, '13/'), 1e-2, 64, 20, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '13/', 20)

# main(check_dir(out_dir, '14/'), 1e-3, 64, 50, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '14/', 50)

# main(check_dir(out_dir, '15/'), 1e-4, 32, 50, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '15/', 50)

# main(check_dir(out_dir, '16/'), 1e-5, 32, 50, 0.5, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '16/', 50)

# main(check_dir(out_dir, '17/'), 1e-3, 32, 50, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '17/', 50)

# main(check_dir(out_dir, '18/'), 1e-3, 32, 20, 0.4, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '18/', 20)

# main(check_dir(out_dir, '19/'), 1e-3, 128, 50, 0.7, 0.7)
# # plot_setting(valid_accuracies, plot_dir, '19/', 50)

# main(check_dir(out_dir, '20/'), 1e-3, 64, 50, 0.7, 0.7)
# # plot_setting(valid_accuracies, plot_dir, '20/', 50)

# main(check_dir(out_dir, '21/'), 1e-3, 64, 50, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '21/', 50)

# main(check_dir(out_dir, '22/'), 1e-4, 64, 50, 0.7)
# # plot_setting(valid_accuracies, plot_dir, '22/', 50)

# main(check_dir(out_dir, '23/'), 1e-4, 64, 50, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '23/', 50)

# main(check_dir(out_dir, '24/'), 1e-4, 128, 50, 0.7)
# # plot_setting(valid_accuracies, plot_dir, '24/', 50)

# main(check_dir(out_dir, '25/'), 1e-4, 128, 50, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '25/', 50)

# main(check_dir(out_dir, '26/'), 1e-4, 128, 50, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '26/', 50)

# main(check_dir(out_dir, '27/'), 1e-3, 128, 30, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '27/', 50)

# main(check_dir(out_dir, '28/'), 0.005, 128, 50, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '28/', 50)

# main(check_dir(out_dir, '29/'), 1e-3, 32, 20, 0.3)
# # plot_setting(valid_accuracies, plot_dir, '29/', 20)

# main(check_dir(out_dir, '30/'), 1e-3, 128, 100, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '30/', 100)

# main(check_dir(out_dir, '31/'), 1e-4, 128, 100, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '31/', 100)

# main(check_dir(out_dir, '32/'), 1e-4, 128, 100, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '32/', 100)

# main(check_dir(out_dir, '33/'), 1e-4, 128, 200, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '33/', 200)

# main(check_dir(out_dir, '34/'), 1e-4, 256, 200, 0.6)
# # plot_setting(valid_accuracies, plot_dir, '34/', 200)

# main(check_dir(out_dir, '35/'), 1e-4, 256, 200, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '35/', 200)

# main(check_dir(out_dir, '36/'), 1e-4, 128, 200, 0.5)
# # plot_setting(valid_accuracies, plot_dir, '36/', 200)

# main(check_dir(out_dir, '37/'), 1e-3, 64, 20, 0.4)
# # plot_setting(valid_accuracies, plot_dir, '37/', 20)

# main(check_dir(out_dir, '38/'), 1e-3, 64, 20, 0.3)
# # plot_setting(valid_accuracies, plot_dir, '38/', 20)

################################################################################

valid_accuracies = sorted(valid_accuracies.items(), key=lambda t: t[1], reverse=True)

print(valid_accuracies)
print(valid_accuracies[:10])

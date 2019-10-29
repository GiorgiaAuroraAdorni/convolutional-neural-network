#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_setting(out_dir, model_dir, epoch):
    dir = out_dir + model_dir

    if not os.path.exists(dir + 'img/'):
        os.makedirs(dir + 'img/')

    set_dir = [dir + 'train.txt',
               dir + 'validation.txt',
               dir + 'test.txt']

    # my_plot(dir, set_dir, 'Loss', epoch)
    my_plot(dir, set_dir, 'Accuracy', epoch)


def my_plot(dir, set_dir, title, epoch):

    with open(set_dir[0], 'r') as f:
        train_lines = f.readlines()
        train_lines = train_lines[1:]

    with open(set_dir[1], 'r') as f:
        valid_lines = f.readlines()
        valid_lines = valid_lines[1:]

    train_loss = np.array([])
    train_accuracy = np.array([])

    valid_loss = np.array([])
    valid_accuracy = np.array([])

    for line in train_lines:
        el = line.strip('\n').split(',')
        train_loss = np.append(train_loss, float(el[2]))
        train_accuracy = np.append(train_accuracy, float(el[3]))

    for line in valid_lines:
        el = line.strip('\n').split(',')
        valid_loss = np.append(valid_loss, float(el[1]))
        valid_accuracy = np.append(valid_accuracy, float(el[2]))

    print(np.max(valid_accuracy))

    x_train = np.arange(1, len(train_loss) * epoch, epoch, dtype=int)
    x_valid = np.arange(1, len(train_loss) * epoch, len(train_loss), dtype=int)

    approx_indices = np.arange(1, len(train_loss), epoch, dtype=int)
    train_approx_x = x_train[approx_indices]

    plt.xlabel('epoch/iteration', fontsize=11)

    if title == 'Accuracy':
        plt.plot(x_train, train_accuracy, label='Train ' + title)
        train_approx_y = train_accuracy[approx_indices]
        plt.plot(train_approx_x, train_approx_y, label='Smoothed Train ' + title)
        plt.plot(x_valid, valid_accuracy, label='Valid ' + title)
        plt.ylabel('accuracy', fontsize=11)
        plt.ylim(0, 1)
    elif title == 'Loss':
        plt.plot(x_train, train_loss, label='Train ' + title)
        train_approx_y = train_loss[approx_indices]
        plt.plot(train_approx_x, train_approx_y, label='Smoothed Train ' + title)
        plt.plot(x_valid, valid_loss, label='Valid ' + title)
        plt.ylabel('loss', fontsize=11)
        plt.ylim(0, 7)

    plt.legend()
    plt.title(title + ' model ' + model_dir.strip('/'), weight='bold', fontsize=12)
    plt.savefig(dir + 'img/' + title + '.png')
    plt.show()


out_dir = 'out/'

### Experiment 1 ###
model_dir = '1/'
plot_setting(out_dir, model_dir, 50)

### Experiment 2 ###w
model_dir = '2/'
plot_setting(out_dir, model_dir, 50)

### Experiment 3 ###w
model_dir = '3/'
plot_setting(out_dir, model_dir, 50)

### Experiment 4 ###w
model_dir = '4/'
plot_setting(out_dir, model_dir, 50)

### Experiment 5 ###w
model_dir = '5/'
plot_setting(out_dir, model_dir, 50)

### Experiment 6 ###w
model_dir = '6/'
plot_setting(out_dir, model_dir, 20)

### Experiment 7 ###w
model_dir = '7/'
plot_setting(out_dir, model_dir, 20)

### Experiment 8 ###w
# model_dir = '8/'
# plot_setting(out_dir, model_dir, 50)

### Experiment 9 ###w
model_dir = '9/'
plot_setting(out_dir, model_dir, 50)

### Experiment 10 ###w
model_dir = '10/'
plot_setting(out_dir, model_dir, 50)

### Experiment 11 ###w
model_dir = '11/'
plot_setting(out_dir, model_dir, 20)

### Experiment 12 ###w
model_dir = '12/'
plot_setting(out_dir, model_dir, 20)

### Experiment 13 ###w
model_dir = '13/'
plot_setting(out_dir, model_dir, 20)

### Experiment 14 ###w
model_dir = '14/'
plot_setting(out_dir, model_dir, 20)

### Experiment 15 ###w
model_dir = '15/'
plot_setting(out_dir, model_dir, 50)

### Experiment 16 ###w
model_dir = '16/'
plot_setting(out_dir, model_dir, 50)

### Experiment 17 ###w
model_dir = '17/'
plot_setting(out_dir, model_dir, 50)

### Experiment 18 ###w
model_dir = '18/'
plot_setting(out_dir, model_dir, 50)

### Experiment 19 ###w
model_dir = '19/'
plot_setting(out_dir, model_dir, 50)

### Experiment 20 ###w
model_dir = '20/'
plot_setting(out_dir, model_dir, 50)

### Experiment 21 ###w
model_dir = '21/'
plot_setting(out_dir, model_dir, 50)

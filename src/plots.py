#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_setting(out_dir, model_dir, epoch):
    dir = out_dir + model_dir

    set_dir = [dir + 'train.txt',
               dir + 'validation.txt',
               dir + 'test.txt']

    img_dir = out_dir + 'img/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_dir = img_dir + model_dir.strip('/') + '-'

    # my_plot(set_dir, img_dir, 'Loss', epoch)
    my_plot(set_dir, img_dir, 'Accuracy', epoch)


def my_plot(set_dir, img_dir, title, epoch):

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
        train_loss = np.append(train_loss, float(el[2]))
        train_accuracy = np.append(train_accuracy, float(el[3]))

    for line in valid_lines:
        el = line.strip('\n').split(',')
        valid_loss = np.append(valid_loss, float(el[1]))
        valid_accuracy = np.append(valid_accuracy, float(el[2]))

    print(set_dir[1], valid_accuracy[-1])

    approx_indices = np.arange(1, len(train_loss), len(train_loss) / epoch, dtype=int)
    x = np.arange(1, epoch + 1, dtype=int)

    plt.xlabel('epoch/iteration', fontsize=11)

    if title == 'Accuracy':
        train_accuracy = train_accuracy[approx_indices]
        plt.plot(x, train_accuracy, label='Train ' + title)
        plt.plot(x, valid_accuracy, label='Valid ' + title)
        plt.ylabel('accuracy', fontsize=11)
        plt.ylim(0, 1)

    elif title == 'Loss':
        train_loss = train_loss[approx_indices]
        plt.plot(x, train_loss, label='Train ' + title)
        plt.plot(x, valid_loss, label='Valid ' + title)
        plt.ylabel('loss', fontsize=11)
        plt.ylim(0, 7)

    plt.legend()
    plt.title(title + ' model ' + model_dir.strip('/'), weight='bold', fontsize=12)
    plt.savefig(img_dir + title + '.png')
    plt.show()


########################################################################################################################
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

# ### Experiment 20 ###w
# model_dir = '20/'
# plot_setting(out_dir, model_dir, 50)
#
# ### Experiment 21 ###w
# model_dir = '21/'
# plot_setting(out_dir, model_dir, 50)
#
# ### Experiment 22 ###w
# model_dir = '22/'
# plot_setting(out_dir, model_dir, 50)
#
# ### Experiment 23 ###w
# model_dir = '23/'
# plot_setting(out_dir, model_dir, 50)
#
# ### Experiment 24 ###w
# model_dir = '24/'
# plot_setting(out_dir, model_dir, 50)
#
# ### Experiment 25 ###w
# model_dir = '25/'
# plot_setting(out_dir, model_dir, 50)
### Experiment 26 ###w
model_dir = '26/'
plot_setting(out_dir, model_dir, 50)

### Experiment 27 ###w
model_dir = '27/'
plot_setting(out_dir, model_dir, 30)

### Experiment 28 ###w
model_dir = '28/'
plot_setting(out_dir, model_dir, 50)

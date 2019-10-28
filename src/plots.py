#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os

def my_plot(set_dir, title, is_training):

    with open(set_dir[0], 'r') as f:
        lines = f.readlines()
        lines = lines[1:]

    epoch = []
    if is_training:
        iteration = []
    loss = []
    accuracy = []

    for line in lines:
        el = line.strip('\n').split(',')

        epoch.append(el[0])
        if is_training:
            iteration.append(el[1])
        loss.append(float(el[2]))
        accuracy.append(float(el[3]))

    if is_training:
        plt.xlabel('iteration', fontsize=11)
        if title == 'Accuracy':
            plt.plot(iteration, accuracy, label='Train ' + title)
            plt.ylabel('accuracy', fontsize=11)
        elif title == 'Loss':
            plt.plot(iteration, loss, label='Train ' + title)
            plt.ylabel('loss', fontsize=11)
    else:
        plt.xlabel('epoch', fontsize=11)
        if title == 'Accuracy':
            plt.plot(epoch, accuracy, label='Validation ' + title)
            plt.ylabel('accuracy', fontsize=11)
        elif title == 'Loss':
            plt.plot(epoch, loss, label='Validation ' + title)
            plt.ylabel('loss', fontsize=11)

    plt.title(title, weight='bold', fontsize=12)
    plt.savefig(dir + 'img/' + title + '.png')
    plt.show()


out_dir = 'out/'
model_dir = '1/'
dir = out_dir + model_dir


if not os.path.exists(dir + 'img/'):
    os.makedirs(dir + 'img/')

set_dir = [dir + 'train.txt',
           dir + 'validation.txt',
           dir + 'test.txt']


my_plot(set_dir, 'Loss', True)
my_plot(set_dir, 'Accuracy', True)

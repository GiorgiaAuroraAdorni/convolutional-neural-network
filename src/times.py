#!/usr/bin/env python3

import numpy as np

plot_dir = 'out/'

model_dir = '3/'

dir = plot_dir + model_dir
set_dir = [dir + 'train.txt',
            dir + 'validation.txt',
            dir + 'test.txt']

with open(set_dir[0], 'r') as f:
    train_lines1 = f.readlines()[1:]

with open(set_dir[1], 'r') as f:
    valid_lines1 = f.readlines()[1:]


train_time = np.array([])
valid_time = np.array([])
train_loss = np.array([])
valid_loss = np.array([])
train_accuracy = np.array([])
valid_accuracy = np.array([])

for line in train_lines1:
    el = line.strip('\n').split(',')
    train_time = np.append(train_time, float(el[3]))
    train_loss = np.append(train_loss, float(el[1]))
    train_accuracy = np.append(train_accuracy, float(el[2]))

for line in valid_lines1:
    el = line.strip('\n').split(',')
    valid_time = np.append(valid_time, float(el[3]))
    valid_loss = np.append(valid_loss, float(el[1]))
    valid_accuracy = np.append(valid_accuracy, float(el[2]))

total_train_time = np.sum(train_time)
total_valid_time = np.sum(valid_time)


print('total_train_time: ', total_train_time, total_train_time / train_time.size)
print('total_valid_time: ', total_valid_time, total_valid_time / valid_time.size)
print('train_loss: ', train_loss[-1], 'valid_loss: ', valid_loss[-1])
print('train_accuracy: ', train_accuracy[-1], 'valid_accuracy: ', valid_accuracy[-1])

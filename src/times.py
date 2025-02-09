#!/usr/bin/env python3

import numpy as np

plot_dir = 'out/'

model_dirs = ['1/', '2/', '3/', '3b/', '3c/', '4/', '4b/', '5/', '5b/', '5c/', '6/',
              '7/', '8/', '8b/', '8c/', '9/', '9b/', '9c/', '10/', '10c/']

for model_dir in model_dirs:
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

    print(model_dir)
    print('total_train_time: ', total_train_time)
    # print('total_valid_time: ', total_valid_time, total_valid_time / valid_time.size)
    print('train_loss: ', train_loss[-1], 'valid_loss: ', valid_loss[-1])
    print('train_accuracy {:.2f}%:  valid_accuracy {:.2f}%: '.format(train_accuracy[-1] * 100, valid_accuracy[-1] * 100))

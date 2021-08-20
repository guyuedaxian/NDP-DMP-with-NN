#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time        : 2021/6/16 3:47 下午
# @Author      : linksdl
# @ProjectName : acs-project-msc_project_ndp
# @File        : train_ndp.py


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from datetime import datetime
import os
import argparse
import json
from ndp.ndp_nets.scnn_ndp_main import NdpSCNN
from imednet.data.smnist_loader import MatLoader, Mapping

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='scnn-ndp-il')
args = parser.parse_args()


# training on different synthetic dataset
dataset_name = 'smnist-mb' # smnist, smnist-awgn, smnist-mb, smnist-rc-awgn
neuros = 500   # 200 or 500
if dataset_name == 'smnist':
    data_path = '../../imednet/data/s-mnist/40x40-smnist.mat'
    data_stub = 'smnist'
    pre_trained = '../mnist_cnn/cnn_trained/mnist_cnn_net_simple_' + str(neuros) +'(mnist).pt'

elif dataset_name == 'smnist-awgn':
    data_path = '../../imednet/data/s-mnist/40x40-smnist-with-awgn.mat'
    data_stub = 'smnist_awgn_9_5_snr'
    pre_trained = '../mnist_cnn/cnn_trained/mnist_cnn_net_simple_' + str(neuros) + '(mnist-awgn).pt'

elif dataset_name == 'smnist-mb':
    data_path = '../../imednet/data/s-mnist/40x40-smnist-with-motion-blur.mat'
    data_stub = 'smnist_mb'
    pre_trained = '../mnist_cnn/cnn_trained/mnist_cnn_net_simple_' + str(neuros) + '(mnist-motion-blur).pt'

elif dataset_name == 'smnist-rc-awgn':
    data_path = '../../imednet/data/s-mnist/40x40-smnist-with-reduced-contrast-and-awgn.mat'
    data_stub = 'smnist_rc_awgn'
    pre_trained = '../mnist_cnn/cnn_trained/mnist_cnn_net_simple_' + str(neuros) + '(mnist-reduced-contrast-and-awgn).pt'

images1, outputs, scale, or_tr = MatLoader.load_data(data_path,load_original_trajectories=True)

images = np.array([cv2.resize(img, (28, 28)) for img in images1]) / 255.0
input_size = images.shape[1] * images.shape[2]

inds = np.arange(12000)
np.random.shuffle(inds)
test_inds = inds[10000:]
train_inds = inds[:10000]
X = torch.Tensor(images[:12000]).float() # [12000, 28, 28]
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]

time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = '../ndp_models/' + args.name

# hyper parameters
k = 1
T = 300 / k
N = 25
learning_rate = 0.001
num_epochs = 150
batch_size = 100

param_str = "T" + str(T) + "_K" + str(k) + "_N" + str(N) + "_L" + str(learning_rate) + "_E" + str(num_epochs) + "_B" + str(batch_size)
model_save_path = model_save_path + '_' + '(' + dataset_name + ')_(' + param_str + ')_(' + time + ')'
os.mkdir(model_save_path)

image_save_path = model_save_path + '/images'
os.mkdir(image_save_path)

# data sets
Y = Y[:, ::k, :]
X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]

# the ndpcnn model
ndpn = NdpSCNN(T=T, l=1, N=N, pt=pre_trained, state_index=np.arange(2))
optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)

loss_values = []
# training process
for epoch in range(num_epochs):
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    for ind in np.split(inds, len(inds) // batch_size):
        optimizer.zero_grad()
        # ndpn output y
        y_h = ndpn(X_train[ind], Y_train[ind, 0, :]) # [100, 301, 2]
        loss = torch.mean((y_h - Y_train[ind]) ** 2)
        loss.backward()
        optimizer.step()

    # exampes for 0-9
    test_sample_indices = [490, 901, 2732, 1623, 214, 1715, 1976, 977, 988, 1629]
    y_h = ndpn(X[test_sample_indices], Y[test_sample_indices, 0, :])
    y_r = Y[test_sample_indices]
    for i in range(0, len(test_sample_indices)):
        plt.figure()
        image = images1[test_sample_indices[i]]
        H, W = image.shape
        plt.imshow(image, cmap='gray', extent=[0, H + 1, W + 1, 0])
        plt.plot(y_h[i, :, 0].detach().cpu().numpy(), y_h[i, :, 1].detach().cpu().numpy(), c='r', linewidth=3)
        plt.plot(y_r[i, :, 0].detach().cpu().numpy(), y_r[i, :, 1].detach().cpu().numpy(), c='b', linewidth=3)
        plt.axis('on')
        plt.savefig(image_save_path + '/valid_img_' + str(epoch) + '_' + str(i) + '.png')

    for i in range(0, len(test_sample_indices)):
        plt.figure()
        image = images1[test_sample_indices[i]]
        H, W = image.shape
        plt.imshow(image, cmap='gray', extent=[0, H + 1, W + 1, 0])
        plt.plot(y_h[i, :, 0].detach().cpu().numpy(), y_h[i, :, 1].detach().cpu().numpy(), c='r', linewidth=3)
        plt.axis('on')
        plt.savefig(image_save_path + '/valid_img_r_' + str(epoch) + '_' + str(i) + '.png')

    # if epoch % 2 == 0:
    x_test = X_test[np.arange(100)]
    y_test = Y_test[np.arange(100)]
    y_htest = ndpn(x_test, y_test[:, 0, :])
    for j in range(10):
        plt.figure(figsize=(8, 8))
        plt.plot(0.667 * y_h[j, :, 0].detach().cpu().numpy(), -0.667 * y_h[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
        plt.axis('off')
        plt.savefig(image_save_path + '/train_img_' + str(epoch) + '_' + str(j) + '.png')

        plt.figure(figsize=(8, 8))
        img = X_train[ind][j].cpu().numpy() * 255
        img = np.asarray(img * 255, dtype=np.uint8)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(image_save_path + '/ground_train_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        plt.figure(figsize=(8, 8))
        plt.plot(0.667 * y_htest[j, :, 0].detach().cpu().numpy(), -0.667 * y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
        plt.axis('off')
        plt.savefig(image_save_path + '/test_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        plt.figure(figsize=(8, 8))
        img = X_test[j].cpu().numpy() * 255
        img = np.asarray(img * 255, dtype=np.uint8)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(image_save_path + '/ground_test_img_' + str(epoch) + '_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

    test = ((y_htest - y_test) ** 2).mean(1).mean(1)
    # loss = torch.mean((y_h - Y_train[ind]) ** 2) loss value
    print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
    loss_values.append(str(test.mean().item()))
    torch.save(ndpn.state_dict(), model_save_path + '/scnn-model.pt')

# write value to file
with open(model_save_path + '/test_loss.txt', 'w') as f:
    f.write(json.dumps(loss_values))

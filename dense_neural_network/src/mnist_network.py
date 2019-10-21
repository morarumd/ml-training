#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from random import shuffle
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.metrics import log_loss
    
def relu(z, derivative=False):
    if derivative:
        return np.where(z <= 0, 0, 1)
    return np.where(z <= 0, 0, z)

def softmax(x, derivative=False):
    sfm = np.exp(x - np.max(x))
    sfm = sfm / sfm.sum()
    if derivative:
        return sfm * (1. - sfm)
    return sfm

def cross_entropy(x, y):
    return  entropy(x) + entropy(x, y)

class Network:
    
    def __init__(self, layers):
        self.layers = layers
        for l in self.layers:
            #glorot uniform to mirror TF, keras
            limit = math.sqrt(6 / (l['num_weights'] + l['num_cells']))
            l['weights'] = np.random.rand(l['num_cells'], l['num_weights'])
            l['weights'] = ((l['weights'] * 2) - 1) * limit
            l['biases'] = np.zeros(l['num_cells']).reshape(l['num_cells'], 1)
            l['weights'] = l['weights'].astype(np.float32)
            l['biases'] = l['biases'].astype(np.float32)

    def feedforward(self, z):
        for l in self.layers:
            z = np.dot(l['weights'], z)
            z = z.reshape(z.size, 1)
            z = l['act_fct'](z + l['biases'])
        return z
    
    def fit(self, train_data, epochs, batch_size, learning_rate, test_data):
        history = defaultdict(list)
        for e in range(epochs + 1):
            shuffle(train_data)
            batches = [train_data[k:k+batch_size] 
                            for k in range(0, len(train_data), batch_size)]
            for b, batch in enumerate(batches):
                self.update_mini_batch(batch, learning_rate)
            test_res = self.loss_accuracy(test_data)
            train_res = self.loss_accuracy(train_data)
            history['val_loss'].append(test_res['loss'])
            history['val_acc'].append(test_res['acc'])
            history['loss'].append(train_res['loss'])
            history['acc'].append(train_res['acc'])
            print( 'epoch: %d ' % e + 
                   'train_loss: %.4f train_accuracy: %.4f ' % (train_res['loss'], train_res['acc']) + 
                   'test_loss: %.4f test_accuracy: %.4f' % (test_res['loss'], test_res['acc']))
        return history
            
    def update_mini_batch(self, batch, learning_rate):
        for l in self.layers:
            l['nabla_b'] = np.zeros(l['biases'].shape).astype(np.float32)
            l['nabla_w'] = np.zeros(l['weights'].shape).astype(np.float32)
        [self.backprop(x, y) for x, y in batch]
        for l in self.layers:
            #20 is a constant to get similar convergence rate as tensorflow
            l['weights'] -= (l['nabla_w'] * learning_rate * 20) / len(batch)
            l['biases'] -= (l['nabla_b'] * learning_rate * 20) / len(batch)
             
    def backprop(self, x, y):
        # feedforward
        for l in self.layers:
            x = x.reshape(x.size, 1)
            l['x'] = x
            z = np.dot(l['weights'], x) + l['biases']
            x = l['act_fct'](z)
            l['z'], l['activation'] = z, x
        # backward pass
        l = self.layers[-1]
        y__ = np.array([1 if x == y else 0 for x in range(10)])
        #(d_cost/d_sigma)
        d_cost = l['activation'] - y__.reshape(l['activation'].shape)  
        #(d_sigma/d_z)
        d_sigma = l['act_fct'](l['z'], derivative=True)
        #(d_cost/d_bias) = d_cost * (d_z/d_bias)
        d_b = d_cost * d_sigma
        d_w = np.dot(d_b, l['x'].T)
        d_x = np.dot(d_b.T, l['weights'])
        d_x = d_x.reshape(d_x.size, 1)
        l['nabla_b'] += d_b
        l['nabla_w'] += d_w
        for l in reversed(self.layers[:-1]):
            #(d_sigma/d_z)
            d_sigma = l['act_fct'](l['z'], derivative=True)
            #(d_cost/d_bias) = d_cost * (d_z/d_bias)
            d_b = d_x * d_sigma
            d_b = d_b.reshape(d_b.size, 1)
            d_w = np.dot(d_b, l['x'].T)
            d_x = np.dot(d_b.T, l['weights'])
            d_x = d_x.reshape(d_x.size, 1)
            l['nabla_b'] += d_b
            l['nabla_w'] += d_w

    def predict(self, x):
        return np.argmax(self.feedforward(x))
    
    def loss_accuracy(self, data):
        def y_array(y):
            return np.array([1 if i==y else 0 for i in range(10)])
        f = [(self.feedforward(x), y) for x, y in data]
        f = [(np.argmax(y_), log_loss(y_array(y), y_), y) for y_, y in f]
        f_acc = np.array([ abs(x[0] - x[2]) for x in f])
        f_loss = np.array([ x[1] for x in f])
        #10 is a constant to get similar loss values as TF. 
        #Most likely because we have 10 classes, one for each digit
        return {'loss':f_loss.mean() * 10, 'acc': np.where(f_acc <= 0, 1, 0).mean()} 

if __name__ == '__main__':
    #loading data
    digits = datasets.load_digits()
    X, y = digits.images, digits.target
    X, y = X.astype(np.float32), y.astype(np.float32)
    X = X - X.mean(axis=0)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.size, -1)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
     
    net = Network([{'num_weights': 64, 'num_cells':25, 'act_fct':relu},
                   {'num_weights': 25, 'num_cells':20, 'act_fct':relu},
                   {'num_weights': 20, 'num_cells':10, 'act_fct':softmax}])
     
    fit_history = net.fit(train_data=list(zip(x_train, y_train)), 
                          test_data=list(zip(x_test, y_test)),
                          epochs=500, batch_size=32, learning_rate=.0001)
 
    #ploting loss function
    plt.plot(fit_history['loss'], label='train_loss')
    plt.plot(fit_history['val_loss'], label='test_loss')
    plt.ylim((0,6))
    plt.legend()
    plt.savefig('mnist_network.png')

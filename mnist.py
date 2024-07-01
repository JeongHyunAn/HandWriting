#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import numpy as np

img_size = 784

def _load_label(filename):
    labels = np.loadtxt(filename, dtype=np.uint8, delimiter=',')
    
    return labels

def _load_img(filename):
    data = np.loadtxt(filename, dtype=np.uint8, delimiter=',')
    data = data.reshape(-1, img_size)
    
    return data

def _convert_numpy(train_image, train_label, test_image):
    dataset = {}
    dataset['train_img'] =  _load_img(train_image)
    dataset['train_label'] =  _load_label(train_label)
    dataset['test_img'] =  _load_img(test_image)
    # dataset['test_label'] =  _load_label('test_label.csv')
    
    return dataset

'''
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img('train_image.csv')
    dataset['train_label'] =  _load_label('train_label.csv')
    dataset['test_img'] =  _load_img('test_image.csv')
    # dataset['test_label'] =  _load_label('test_label.csv')
    
    return dataset
'''
def init_mnist(train_image, train_label, test_image):
    dataset = _convert_numpy(train_image, train_label, test_image)
    return dataset

'''
def init_mnist():
    dataset = _convert_numpy()
    return dataset
'''

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    
# def load_mnist(normalize=True, flatten=True, one_hot_label=False):
def load_mnist(train_image, train_label, test_image, normalize=True, flatten=True, one_hot_label=False):

    dataset = init_mnist(train_image, train_label, test_image)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        # dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['train_label'])
    # return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist(train_image, train_label, test_image)


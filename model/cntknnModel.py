import matplotlib.pyplot as plt 

import numpy as np 
from sklearn.datasets import fetch_openml
import random 

import cntk.tests.test_utils
from sklearn.preprocessing import OneHotEncoder

import cntk as C 

num_samples = 60000
batch_size = 64 
learning_rate = 0.1 

class Batch_Reader(object):
    def __init__(self, data, label):
        self.data = data 
        self.label = label
        self.num_sample = data.shape[0]

    def next_batch(self, batch_size):
        index = random.sample(range(self.num_sample), batch_size)
        return self.data[index,:].astype(float),self.label[index,:].astype(float)
    

mnist = fetch_openml('mnist_784')

train_data = mnist.data[:num_samples,:]
train_label = mnist.target[:num_samples]
test_data = mnist.data[num_samples:,:]
test_label = mnist.target[num_samples:]

enc = OneHotEncoder()

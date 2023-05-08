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


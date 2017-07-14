import ttflow_rts
import tensorflow as tf
import numpy as np
import ttflow_example
import os

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = ttflow_example.mkModel()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train_generator():
    for _ in range(1000):
        yield mnist.train.next_batch(100)

def valid_generator():
    return []

sess = tf.Session()

ttflow_rts.train(sess,model,tf.train.AdamOptimizer(1e-4),train_generator,valid_generator,10)

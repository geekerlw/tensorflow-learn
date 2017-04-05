# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import input_data

#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = input_data.read_data_sets(one_hot=True)
#mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 20]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
#x = tf.placeholder(tf.float32, [None, 784])
# correct answers will go here
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)




# reshape the output from the third convolution for the fully connected layer
# yy = tf.reshape(y3, shape=[-1, 7 * 7 * 200])

#y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
#Ylogits = tf.matmul(y4, w5) + b5
#y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
#cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
# train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(10000):
	batch_x, batch_y = mnist.train.next_batch(100)
	if i % 100 == 0:
		n = np.random.randint(0, 4000)
		accuracy_step = sess.run(accuracy, feed_dict={x: mnist.validation.images[n: (n+100)], y_: mnist.validation.labels[n: (n+100)], keep_prob: 1.0})
		print ("train_step => %d" % i)
		print ("accuracy in this step is: %g" % accuracy_step)
	sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})

# load test_data
test_data = mnist.test.images

predict = tf.argmax(y_conv, 1)
predicted_labels = np.zeros(test_data.shape[0])
for i in range(0, test_data.shape[0]//100):
	predicted_labels[i*100: (i+1)*100] = predict.eval(feed_dict={x: test_data[i*100 : (i+1)*100], keep_prob: 1.0})

np.savetxt('submission.csv',
			np.c_[range(1, len(test_data)+1), predicted_labels],
			delimiter = ',',
			header = 'ImageId,Label',
			comments = '',
			fmt = '%d')

#sess.close()

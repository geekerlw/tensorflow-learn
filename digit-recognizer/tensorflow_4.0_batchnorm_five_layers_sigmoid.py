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

mnist = input_data.read_data_sets(one_hot=True)


# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid+BN)   W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid+BN)   W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid+BN)   W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid+BN)   W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax+BN)   W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]))
O1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
S2 = tf.Variable(tf.ones([M]))
O2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
S3 = tf.Variable(tf.ones([N]))
O3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
S4 = tf.Variable(tf.ones([P]))
O4 = tf.Variable(tf.zeros([P]))
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.zeros([Q]))

## Batch normalisation conclusions with sigmoid activation function:
# BN is applied between logits and the activation function
# On Sigmoids it is very clear that without BN, the sigmoids saturate, with BN, they output
# a clean gaussian distribution of values, especially with high initial learning rates.

# sigmoid, no batch-norm, lr(0.003, 0.0001, 2000) => 97.5%
# sigmoid, batch-norm lr(0.03, 0.0001, 1000) => 98%
# sigmoid, batch-norm, no offsets => 97.3%
# sigmoid, batch-norm, no scales => 98.1% but cannot hold fast learning rate at start
# sigmoid, batch-norm, no scales, no offsets => 96%

# Both scales and offsets are useful with sigmoids.
# With RELUs, the scale variables can be omitted.
# Biases are not useful with batch norm, offsets are to be used instead

# Steady 98.5% accuracy using these parameters:
# moving average decay: 0.998 (equivalent to averaging over two epochs)
# learning rate decay from 0.03 to 0.0001 speed 1000 => max 98.59 at 6500 iterations, 98.54 at 10K it,  98% at 1300it, 98.5% at 3200it

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()

# The model
XX = tf.reshape(X, [-1, 784])

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
Y1 = tf.nn.sigmoid(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
Y2 = tf.nn.sigmoid(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
Y3 = tf.nn.sigmoid(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
Y4 = tf.nn.sigmoid(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(10000):
	batch_x, batch_y = mnist.train.next_batch(100)
	# learning rate decay (with batch norm)
	max_learning_rate = 0.03
	min_learning_rate = 0.0001
	decay_speed = 1000.0
	learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
	if i % 100 == 0:
		accuracy_step = sess.run(accuracy, feed_dict={X: mnist.validation.images, Y_: mnist.validation.labels, tst: True})
		print ("train_step => %d" % i)
		print ("accuracy in this step is: %g" % accuracy_step)
	sess.run(train_step, feed_dict={X: batch_x, Y_: batch_y, lr: learning_rate, tst: False})
	sess.run(update_ema, {X: batch_x, Y_: batch_y, tst: False, iter: i})

# load test_data
test_data = mnist.test.images

predict = tf.argmax(Y, 1)
predicted_labels = np.zeros(test_data.shape[0])
for i in range(0, test_data.shape[0]//100):
	predicted_labels[i*100: (i+1)*100] = predict.eval(feed_dict={X: test_data[i*100 : (i+1)*100], tst: True})

np.savetxt('submission.csv',
			np.c_[range(1, len(test_data)+1), predicted_labels],
			delimiter = ',',
			header = 'ImageId,Label',
			comments = '',
			fmt = '%d')

#sess.close()
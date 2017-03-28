import tensorflow as tf
import numpy as np
import pandas as pd
import input_data

mnist = input_data.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

xx = tf.reshape(x, shape=[-1, 784])
y = tf.nn.softmax(tf.matmul(xx, w) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(20000):
	batch_x, batch_y = mnist.train.next_batch(100)
	
	if i % 100 == 0:
		accuracy_step = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
		print ("train_step => %d" % i)
		print ("accuracy in this step is: %g" % accuracy_step)
	sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# load test_data
test_data = mnist.test.images

predict = tf.argmax(y, 1)
predicted_labels = np.zeros(test_data.shape[0])
for i in range(0, test_data.shape[0]//100):
	predicted_labels[i*100: (i+1)*100] = predict.eval(feed_dict={x: test_data[i*100 : (i+1)*100]})


np.savetxt('submission.csv',
			np.c_[range(1, len(test_data)+1), predicted_labels],
			delimiter = ',',
			header = 'ImageId,Label',
			comments = '',
			fmt = '%d')

sess.close()
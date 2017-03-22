import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#import input_data
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# set input and out put placeholders
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# set weight and bias variables
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# shape the input data
xx = tf.reshape(x, [-1, 784])

# model
y = tf.nn.softmax(tf.matmul(xx, w) + b)

# cross entropy
cross_entropy = -tf.reduce_mean(y_*tf.log(y)) * 1000.0

# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training loop
for i in range(1000):
	batch_x, batch_y = mnist.train.next_batch(100)

	if i % 100 == 0:
		#train_accuracy = sess.accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
		#print ("step %d, training accuracy %g" % (i, train_accuracy))	
		print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

	sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print ("final accuracy: %g" % accuracy )

import numpy as np
import pandas as pd

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def test_images_get(filename):
	# read test data from csv file
	file = pd.read_csv(filename)
	# split and format data
	data = file.iloc[:, 0:].values
	# format data
	data = data.astype(np.float32)
	# convert from [0:255] to [0.0:1.0]
	data = np.multiply(data, 1.0 / 255.0)
	# get nums of data
	image_nums = len(file[[0]].values.ravel())
	image_size = data.shape[1]
	image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
	images = data.reshape(image_nums, image_width, image_height, 1)
	
	return images
	
def train_images_get(filename):
	# read training data from csv file
	file = pd.read_csv(filename)
	# split and format data
	data = file.iloc[:, 1:].values
	# format data
	data = data.astype(np.float32)
	# convert from [0:255] to [0.0:1.0]
	data = np.multiply(data, 1.0 / 255.0)
	# get nums of data
	image_nums = len(file[[0]].values.ravel())
	image_size = data.shape[1]
	image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
	images = data.reshape(image_nums, image_width, image_height, 1)
	
	return images

def train_labels_get(filename, one_hot=False):
	# read training data from csv file
	file = pd.read_csv(filename)
	# get labels
	labels_flat = file[[0]].values.ravel()
	
	# get unique count of labels
	labels_count = np.unique(labels_flat).shape[0]
	if one_hot:
		labels = dense_to_one_hot(labels_flat, labels_count)
		labels = labels.astype(np.uint8)
		return labels
	labels = labels_flat.astype(np.uint8)
	return labels

class dataSet(object):
	def __init__(self, images=None, labels=None):
		#assert images.shape[0] == labels.shape[0], (
		#	"images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		try:
			if len(np.shape(self._labels)) == 1:
				self._labels = dense_to_one_hot(self._labels, len(np.unique(self._labels)))
		except:
			traceback.print_exc()
		self._epochs_completed = 0
		self._index_in_epoch = 0
		
	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed
		
	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

'''
	images output like X[batch, wight, height, deepth]
	labels output like Y[count]
'''
def read_data_sets(one_hot=False):
	class dataSets(object):
		pass
	data_sets = dataSets()
	
	TRAIN_IMAGES = './data/train.csv'
	TESE_IMAGES = './data/test.csv'
	VALIDATION_SIZE = 5000
	
	train_images = train_images_get(TRAIN_IMAGES)
	train_labels = train_labels_get(TRAIN_IMAGES, one_hot=one_hot)
	test_images = test_images_get(TESE_IMAGES)
	
	validation_images = train_images[:VALIDATION_SIZE]
	validation_labels = train_labels[:VALIDATION_SIZE]
	train_images = train_images[VALIDATION_SIZE:]
	train_labels = train_labels[VALIDATION_SIZE:]
	
	data_sets.train = dataSet(train_images, train_labels)
	data_sets.validation = dataSet(validation_images, validation_labels)
	data_sets.test = dataSet(test_images)

	return data_sets
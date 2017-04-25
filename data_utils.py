import numpy as np
import os
from scipy.misc import imread

class DataSet(object):
	def __init__(self, images):

		self._num_examples = images.shape[0]
		self._images = images
		self._index_in_epoch = 0
		self._epochs_completed = 0

	def images(self):
		'''Returns images.'''
		return self._images

	def num_examples(self):
		'''Returns number of images.'''
		return self._num_examples

	def epochs_completed(self):
		'''Returns number of completed epochs.'''
		return self._epochs_completed

	def next_batch(self, batch_size):
		'''Return the next `batch_size` images from the data set.'''
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:

			self._epochs_completed += 1

			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch

		return self._images[start:end]

def read_images(filenames):
	'''Reads images from file names'''
	images = np.zeros([len(filenames), 128, 128, 3], dtype = np.float32)
	for i, file in enumerate(filenames):
		img = imread(file, mode = 'RGB')
		image = img.astype(np.float32)
		image = np.multiply(image, 1.0 / 255.0)
		images[i] = image

	return images

def read_dataset(path):
	'''Creates data set'''
	dirpath, dirnames, filenames = next(os.walk(path))
	images = read_images([os.path.join(dirpath, filename) for filename in filenames])

	perm = np.arange(images.shape[0])
	np.random.shuffle(perm)
	images = images[perm]

	return DataSet(images)

def input_data(train_path):
	return read_dataset(train_path)

if __name__ == '__main__':
	train_ds = input_data('data/frames')
	print 'Shape:', train_ds.images().shape
	print 'Memory size:', train_ds.images().nbytes / (1024.0 * 1024.0), 'MB'
	print 'Batch shape:', train_ds.next_batch(100).shape
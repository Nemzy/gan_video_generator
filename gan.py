import tensorflow as tf
from ops import *


class DeepConvGAN(object):
	'''Deep convolutional generative adversarial network'''
	def __init__(self, latent_dim, batch_size):
		self.latent_dim = latent_dim
		self.batch_size = batch_size

		# let's make generator
		z = tf.truncated_normal(shape = [self.batch_size, self.latent_dim], name = 'z')
		self._generator = self.generator(z)

		# let's make discriminator
		x = tf.placeholder(tf.float32, [self.batch_size, 128, 128, 3])
		self._discriminator = self.discriminator(x)

		# and generator with discriminator on top
		self._generator_with_discriminator = self.generator_with_discriminator()

		# let's define our losses

		# generator loss
		self._generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._generator_with_discriminator,\
																	labels = tf.ones([self.batch_size, 1])))

		# discriminator loss
		self._discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
					logits = tf.concat([self._discriminator, self._generator_with_discriminator], 0),\
					labels = tf.concat([tf.ones([self.batch_size,1]), tf.zeros([self.batch_size,1])], 0))\
					)

		# and we need optimizers
		self.g_optimizer = tf.train.AdamOptimizer(1e-4).minimize(self._generator_loss)
		self.d_optimizer = tf.train.AdamOptimizer(1e-4).minimize(self._discriminator_loss)

		# END

	
	def generator(self, z, reuse = False, trainable = True):
		'''Creates generator'''

		# first fully connected layer tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
		fc1 = dense(z, self.latent_dim, 1024, 'g_fc1', reuse, trainable)
		fc1 = tf.contrib.layers.batch_norm(fc1, epsilon=1e-5)
		fc1 = tf.nn.relu(fc1)

		# second fully connected layer
		fc2 = dense(fc1, 1024, 16*16*64, 'g_fc2', reuse, trainable)
		fc2 = tf.contrib.layers.batch_norm(fc2, epsilon=1e-5)

		# first deconvolutional layer 16x16x64 -> 32x32x32
		fc2_reshaped = tf.nn.relu(tf.reshape(fc2, [-1, 16, 16, 64]))
		deconv1 = conv2d_transpose(fc2_reshaped, [self.batch_size, 32, 32, 32], 'g_deconv1', reuse, trainable)
		deconv1 = tf.contrib.layers.batch_norm(deconv1, epsilon=1e-5)
		deconv1 = tf.nn.relu(deconv1)

		# second deconvolutional layer 32x32x32 -> 64x64x16
		deconv2 = conv2d_transpose(deconv1, [self.batch_size, 64, 64, 16], 'g_deconv2', reuse, trainable)
		deconv2 = tf.contrib.layers.batch_norm(deconv2, epsilon=1e-5)
		deconv2 = tf.nn.relu(deconv2)

		# third deconvolutional layer 64x64x16 -> 128x128x1
		deconv3 = tf.nn.sigmoid(conv2d_transpose(deconv2, [self.batch_size, 128, 128, 3], 'g_deconv3', reuse, trainable))
		
		return deconv3

	def discriminator(self, x, reuse = False, trainable = True):
		'''Creates discriminator'''

		# first convolutional layer 128x128x3 -> 64x64x16
		conv1 = tf.nn.relu(conv2d(x, 3, 16, 'd_conv1', reuse, trainable))

		# second convolutional layer 64x64x16 -> 32x32x32
		conv2 = tf.nn.relu(conv2d(conv1, 16, 32, 'd_conv2', reuse, trainable))

		# third convolutional layer 32x32x32 -> 16x16x64
		conv3 = tf.nn.relu(conv2d(conv2, 32, 64, 'd_conv3', reuse, trainable))

		# first fully connected layer
		conv3_flatten = tf.reshape(conv3, [-1, 16*16*64])
		fc1 = tf.nn.relu(dense(conv3_flatten, 16*16*64, 1024, 'd_fc1', reuse, trainable))

		# second fully connected layer
		fc2 = dense(fc1, 1024, 1, 'd_fc2', reuse, trainable)

		return fc2

	def generator_with_discriminator(self):
		'''Creates discriminator on top of generator'''
		return self.discriminator(self._generator, reuse = True, trainable = False)

	def g_training_step(self, sess):
		sess.run(self.g_optimizer)

	def d_training_step(self, sess, feed_dict):
		sess.run(self.d_optimizer, feed_dict = feed_dict)

	def generator_loss(self, sess):
		return sess.run(self._generator_loss)

	def discriminator_loss(self, sess, feed_dict):
		return sess.run(self._discriminator_loss, feed_dict = feed_dict)

if __name__ == '__main__':

	# test before use
	gan = DeepConvGAN(100, 50)
	# looks good
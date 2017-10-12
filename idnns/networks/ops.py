import tensorflow as tf
import numpy as np
from idnns.networks.utils import _convert_string_dtype


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                      strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def set_value(x, value):
	"""Sets the value of a variable, from a Numpy array.
	# Arguments
		x: Tensor to set to a new value.
		value: Value to set the tensor to, as a Numpy array
			(of the same shape).
	"""
	value = np.asarray(value)
	tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
	if hasattr(x, '_assign_placeholder'):
		assign_placeholder = x._assign_placeholder
		assign_op = x._assign_op
	else:
		assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
		assign_op = x.assign(assign_placeholder)
		x._assign_placeholder = assign_placeholder
		x._assign_op = assign_op
	session = tf.get_default_session()
	session.run(assign_op, feed_dict={assign_placeholder: value})


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def get_scope_variable(name_scope, var, shape=None, initializer=None):
	with tf.variable_scope(name_scope) as scope:
		try:
			v = tf.get_variable(var, shape, initializer=initializer)
		except ValueError:
			scope.reuse_variables()
			v = tf.get_variable(var)
	return v

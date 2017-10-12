import tensorflow as tf
from idnns.networks.ops import *


def multi_layer_perceptron(x, n_input, n_classes, n_hidden_1, n_hidden_2):
	hidden = []
	input = []
	hidden.append(x)
	# Network Parameters
	# n_input = x.shape[0]  # MNIST data input (img shape: 28*28)
	# n_classes = 10  # MNIST total classes (0-9 digits)

	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Hidden layer with RELU activation
	layer_1_input = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1_input)
	input.append(layer_1)
	hidden.append(layer_1)
	# Hidden layer with RELU activation
	layer_2_input = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2_input)
	input.append(layer_2_input)
	hidden.append(layer_2)
	# Output layer with linear activation
	input_y = tf.matmul(layer_2, weights['out']) + biases['out']
	y_output = tf.nn.softmax(input_y)
	input.append(y_output)
	hidden.append(y_output)
	return y_output, hidden, input


def deepnn(x):
	"""deepnn builds the graph for a deep net for classifying digits.
	Args:
	  x: an input tensor with the dimensions (N_examples, 784), where 784 is the
	  number of pixels in a standard MNIST image.
	Returns:
	  A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
	  equal to the logits of classifying the digit into one of 10 classes (the
	  digits 0-9). keep_prob is a scalar placeholder for the probability of
	  dropout.
	"""
	hidden = []
	input = []
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	hidden.append(x)
	# First convolutional layer - maps one grayscale image to 32 feature maps.
	with tf.name_scope('conv1'):
		with tf.name_scope('weights'):
			W_conv1 = weight_variable([5, 5, 1, 32])
			variable_summaries(W_conv1)
		with tf.name_scope('biases'):
			b_conv1 = bias_variable([32])
			variable_summaries(b_conv1)
		with tf.name_scope('activation'):
			input_con1 = conv2d(x_image, W_conv1) + b_conv1
			h_conv1 = tf.nn.relu(input_con1)
			tf.summary.histogram('activations', h_conv1)
		with tf.name_scope('max_pol'):
			# Pooling layer - downsamples by 2X.
			h_pool1 = max_pool_2x2(h_conv1)
		input.append(input_con1)
		hidden.append(h_pool1)
	with tf.name_scope('conv2'):
		# Second convolutional layer -- maps 32 feature maps to 64.
		with tf.name_scope('weights'):
			W_conv2 = weight_variable([5, 5, 32, 64])
			variable_summaries(W_conv2)
		with tf.name_scope('biases'):
			b_conv2 = bias_variable([64])
			variable_summaries(b_conv2)
		with tf.name_scope('activation'):
			input_con2 = conv2d(h_pool1, W_conv2) + b_conv2
			h_conv2 = tf.nn.relu(input_con2)
			tf.summary.histogram('activations', h_conv2)
		with tf.name_scope('max_pol'):
			# Second pooling layer.
			h_pool2 = max_pool_2x2(h_conv2)
		input.append(input_con2)
		hidden.append(h_pool2)
	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('FC1'):
		with tf.name_scope('weights'):
			W_fc1 = weight_variable([7 * 7 * 64, 1024])
			variable_summaries(W_fc1)
		with tf.name_scope('biases'):
			b_fc1 = bias_variable([1024])
			variable_summaries(b_fc1)
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
		with tf.name_scope('activation'):
			input_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
			h_fc1 = tf.nn.relu(input_fc1)
			tf.summary.histogram('activations', h_fc1)

	with tf.name_scope('drouput'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		input.append(input_fc1)
		hidden.append(h_fc1_drop)
	# Map the 1024 features to 10 classes, one for each digit
	with tf.name_scope('FC2'):
		with tf.name_scope('weights'):
			W_fc2 = weight_variable([1024, 10])
			variable_summaries(W_fc2)
		with tf.name_scope('biases'):
			b_fc2 = bias_variable([10])
			variable_summaries(b_fc2)

	input_y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_conv = tf.nn.softmax(input_y_conv)
	input.append(input_y_conv)
	hidden.append(y_conv)
	return y_conv, keep_prob, hidden, input

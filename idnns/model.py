import functools
import tensorflow as tf
import numpy as np
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        #print hasattr(self, attribute)
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)

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
    input =[]
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
    return y_conv, keep_prob, hidden,input


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

def get_scope_variable(name_scope, var, shape=None,initializer = None):
    with tf.variable_scope(name_scope) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v


class Model:
    """A class that represent model of network"""
    def __init__(self,input_size, layerSize, num_of_classes,learning_rate_local = 0.001, save_file = '', activation_function=0, cov_net= False):
        self.covnet =  cov_net
        self.input_size = input_size
        self.layerSize = layerSize
        self.all_layer_sizes = np.copy(layerSize)
        self.all_layer_sizes = np.insert(self.all_layer_sizes, 0, input_size)
        self.num_of_classes = num_of_classes
        self.num_of_layers = len(layerSize)+1
        self.learning_rate_local = learning_rate_local
        self.save_file= save_file
        self.hidden = None
        self.savers = []
        if activation_function ==1:
            self.activation_function = tf.nn.relu
        else:
            self.activation_function = tf.nn.tanh
        self.prediction
        self.optimize
        self.accuracy


    def initilizae_layer(self, name_scope, row_size, col_size,activation_function, last_hidden):
        #Bulid layer of the network with weights and biases
        weights = get_scope_variable(name_scope=name_scope, var="weights",
                                     shape=[row_size, col_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0 / np.sqrt(
                                         float(row_size))))
        biases = get_scope_variable(name_scope=name_scope, var='biases', shape=[col_size],
                                    initializer=tf.constant_initializer(0.0))

        self.weights_all.append(weights)
        self.biases_all.append(biases)
        variable_summaries(weights)
        variable_summaries(biases)
        with tf.variable_scope(name_scope) as scope:
            input = tf.matmul(last_hidden, weights) + biases
            output = activation_function(input, name='output')
        self.inputs.append(input)
        self.hidden.append(output)
        return output

    @property
    def num_of_layers(self):
        return self.num_of_layers

    @property
    def hidden_layers(self):
        """The hidden layers of the netowrk"""
        if self.hidden is None:
            self.hidden,self.inputs,self.weights_all,self.biases_all = [], [], [],[]
            last_hidden = self.x
            if self.covnet:
                y_conv, self._drouput, self.hidden,self.inputs = deepnn(self.x)
            else:
                self._drouput ='dr'
                #self.hidden.append(self.x)
                for i in xrange(1, len(self.all_layer_sizes)):
                    name_scope = 'hidden' + str(i-1)
                    row_size, col_size = self.all_layer_sizes[i - 1], self.all_layer_sizes[i]
                    activation_function = self.activation_function
                    last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function, last_hidden)
                name_scope = 'final_layer'
                row_size, col_size = self.layerSize[-1], self.num_of_classes
                activation_function = tf.nn.softmax
                last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function, last_hidden)
        return self.hidden

    @lazy_property
    def prediction(self):
        logits = self.hidden_layers[-1]
        return logits

    @lazy_property
    def drouput(self):
        return self._drouput
    @property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_local).minimize(self.cross_entropy)

        return optimizer

    @lazy_property
    def x(self):
        return tf.placeholder(tf.float32, shape=[None, self.input_size], name='x')

    @lazy_property
    def labels(self):
        return tf.placeholder(tf.float32, shape=[None, self.num_of_classes], name='y_true')


    @lazy_property
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @lazy_property
    def cross_entropy(self):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(tf.clip_by_value(self.prediction,1e-20, 1.0 )), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy

    @property
    def save_file(self):
        return self.save_file

    def inference(self, data):
        """Return the predication of the network with the given data"""
        with tf.Session() as sess:
            self.saver.restore(sess, './' + self.save_file)
            feed_dict = {self.x: data}
            pred = sess.run(self.prediction, feed_dict=feed_dict)
        return pred

    def inference_default(self, data):
        session = tf.get_default_session()
        feed_dict = {self.x: data}
        pred = session.run(self.prediction, feed_dict=feed_dict)
        return pred

    def get_layer_with_inference(self, data, layer_index, epoch_index):
        """Return the layer activation's values with the results of the network"""
        with tf.Session() as sess:
            self.savers[epoch_index].restore(sess, './' + self.save_file +str(epoch_index))
            feed_dict = {self.hidden_layers[layer_index]: data[:,0:self.hidden_layers[layer_index]._shape[1]]}
            pred, layer_values = sess.run([self.prediction, self.hidden_layers[layer_index]], feed_dict=feed_dict)
        return pred, layer_values

    def calc_layer_values(self,X, layer_index):
        """Return the layer's values"""
        with tf.Session() as sess:
            self.savers[-1].restore(sess, './' + self.save_file)
            feed_dict = {self.x: X}
            layer_values = sess.run(self.hidden_layers[layer_index], feed_dict=feed_dict)
        return layer_values

    def update_weights_and_calc_values_temp(self, d_w_i_j, layer_to_perturbe, i,j,X):
        """Update the weights of the given layer cacl the output and return it to the original values"""
        if layer_to_perturbe+1 >= len(self.hidden_layers):
            scope_name ='softmax_linear'
        else:
            scope_name = "hidden" + str(layer_to_perturbe)
        weights = get_scope_variable(scope_name, "weights", shape=None, initializer=None)
        session = tf.get_default_session()
        weights_values = weights.eval(session=session)
        weights_values_pert = weights_values
        weights_values_pert[i,j] += d_w_i_j
        set_value(weights, weights_values_pert)
        feed_dict = {self.x: X}
        layer_values = session.run(self.hidden_layers[layer_to_perturbe], feed_dict=feed_dict)
        set_value(weights, weights_values)
        return layer_values

    def update_weights(self, d_w0, layer_to_perturbe):
        """Update the weights' values of the given layer"""
        weights = get_scope_variable("hidden" +str(layer_to_perturbe), "weights", shape=None, initializer=None)
        session = tf.get_default_session()
        weights_values = weights.eval(session=session )
        set_value(weights, weights_values +d_w0)

    def get_wights_size(self, layer_to_perturbe):
        """Return the size of the given layer"""
        weights = get_scope_variable("hidden" + str(layer_to_perturbe), "weights", shape=None, initializer=None)
        return weights._initial_value.shape[1].value, weights._initial_value.shape[0].value

    def get_layer_input(self, layer_to_perturbe,X):
        """Return the input of the given layer for the given data"""
        session = tf.get_default_session()
        inputs = self.inputs[layer_to_perturbe]
        feed_dict = {self.x: X}
        layer_values = session.run(inputs, feed_dict=feed_dict)
        return layer_values



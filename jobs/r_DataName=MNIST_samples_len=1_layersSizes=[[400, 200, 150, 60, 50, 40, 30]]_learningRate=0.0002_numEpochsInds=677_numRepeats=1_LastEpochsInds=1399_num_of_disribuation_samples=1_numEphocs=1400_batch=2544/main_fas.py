import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cPickle
import os
from joblib import Parallel, delayed
import multiprocessing
from numpy import linalg as LA
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import re
import shutil
import entropy_estimators as ee
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
NUM_CORES = multiprocessing.cpu_count()
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def KL(a, b):
    """Calculate the Kullback Leibler divergence between a and b """
    D_KL = np.nansum(np.multiply(a, np.log(np.divide(a, b+np.spacing(1)))))
    return D_KL


def IB_iteration(p_t_x, PXs, PYgivenXs, beta):
    """Calculate Information buttleneack untill converged"""
    for i in range(0,20):
        pts = np.dot(p_t_x,PXs)
        ProbYGivenT_b = np.multiply(PYgivenXs, np.tile(PXs, (PYgivenXs.shape[0], 1)))
        PYgivenTs_update = np.dot(ProbYGivenT_b, np.multiply(p_t_x, np.tile(1. / (pts), (p_t_x.shape[1], 1)).T).T)
        d1 = np.tile(np.nansum(np.multiply(PYgivenXs, np.log(PYgivenXs)), axis=0), (p_t_x.shape[0], 1))
        d2 = np.dot(-np.log(PYgivenTs_update.T + np.spacing(1)), PYgivenXs)
        DKL = d1 + d2
        probTgivenXs = np.exp(-beta * (DKL)) * pts[:, np.newaxis]
        probTgivenXs = probTgivenXs / np.tile(np.nansum(probTgivenXs, axis=0), (probTgivenXs.shape[0], 1))
    return probTgivenXs,PYgivenTs_update


def mean_DKL(beta, p_t_x, PXs, PYgivenXs):
    """Return The mean D_KL between p(x|t) and p(x|t) based on the IB"""
    probTgivenXs, PYgivenTs_update = IB_iteration(p_t_x, PXs, PYgivenXs, beta)
    D_KL = np.mean([KL(p_x_given_t_new, px_given_t) for p_x_given_t_new, px_given_t in zip(probTgivenXs.T, p_t_x.T)])
    return D_KL


def calc_information(probTgivenXs, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT


def process_and_calc_information_inner(PXgivenTs, PYgivenTs, PTs, PYs, PXs, PYgivenXs):
    """Calculate I(X;T) and I(Y;T) """
    PXgivenTs ,PYgivenTs= np.vstack(PXgivenTs).T,np.vstack(PYgivenTs).T
    PXs,PYs = np.asarray(PXs).T,np.asarray(PYs).T
    PTs = np.asarray(PTs, dtype=np.float64).T
    PTgivenXs_not_divide =np.multiply(PXgivenTs,np.tile(PTs,(PXgivenTs.shape[0],1))).T
    PTgivenXs = np.multiply(PTgivenXs_not_divide, np.tile((1./(PXs)), (PXgivenTs.shape[1], 1)))
    ITX, IYT = calc_information(PTgivenXs, PYgivenTs, PXs, PYs)
    return ITX, IYT


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
    """Calculate the p(x|T) and p(y|T)"""
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
    return p_x_ts, p_y_ts


def estimate_Information(Xs, Ys, Ts):
    """Estimation of the MI from mising data"""
    estimate_IXT = ee.midc(Xs, Ts)
    estimate_IYT = ee.mi(Ys, Ts)
    return estimate_IXT, estimate_IYT


def calc_infomration_for_layer(data, bins, label, b, b1, len_unique_a, pys, pxs, py_x):
    """Calculate I(T;Y) and I(X;T) based on the givin data and labels for one layer"""
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    #pxys[:, 0] is  p(x|T) and pxys[:, 1] is p(y|T)
    pxys = np.array(
        [calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
    )
    local_IXT, local_ITY = process_and_calc_information_inner(pxys[:, 0], pxys[:, 1], p_ts, pys, pxs, py_x)
    #est_IXT, est_ITY = estimateInformation(Xs,Ys, Ts )
    return local_IXT, local_ITY


def calc_infomration_all_neruons(data, bins, label, b, b1, len_unique_a, pys, pxs, py_x):
    """Calculate the information for all the neuron in the layer"""
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    uni_array =np.array([np.unique(neuron_dist, return_index=False, return_inverse=True, return_counts=True)
        for neuron_dist in digitized.T])
    unique_array, unique_inverse_t, unique_counts  = uni_array[:,0],uni_array[:,1] ,uni_array[:,2]
    p_ts =np.array([ unique_counts_per_neuron / float(sum(unique_counts_per_neuron))
        for unique_counts_per_neuron in unique_counts])
    I_XT, I_TY = [],[]
    for i_n in range(len(p_ts)):
        current_unique_array = unique_array[i_n]
        current_unique_inverse_t = unique_inverse_t[i_n]
        pxys = np.array(
            [calc_probs(i, current_unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(current_unique_array))]
        )
        local_IXT, local_ITY = process_and_calc_information_inner(pxys[:, 0], pxys[:, 1], p_ts[i_n], pys, pxs, py_x)
        I_XT.append(local_IXT)
        I_TY.append(local_ITY)
    return np.array(I_XT), np.array(I_TY)


def get_information_for_epoch(iter_index, ws_iter_index, bins, label, b, b1, len_unique_a, pys, pxs, py_x):
    """Calculate the information for all the layers for specific epoch"""
    print ('Calculating epoch number - ', iter_index)
    iter_infomration = np.array(
        [calc_infomration_for_layer(ws_iter_index[i], bins, label, b, b1, len_unique_a, pys, pxs, py_x) for i in range(0, len(ws_iter_index))])
    #iter_infomrationNeurons = np.array([calc_infomration_for_layer(ws_iter_index[i], bins, unique_inverse, label, b, b1, len_unique_a, pys, pxs, py_x,
    #                     ) for i in range(0, len(ws_iter_index))])
    return iter_infomration


def get_infomration(ws, x, label,num_of_bins):
    """Calculate the information for the network"""
    print label.shape
    label = np.array(label).astype(np.float)
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    p_ys = np.sum(label, axis=0) / float(label.shape[0])
    p_xs = unique_counts / float(sum(unique_counts))
    bins = np.linspace(-1, 1, num_of_bins)
    p_y_given_x = [label[unique_inverse == i] for i in range(0, len(unique_array))]
    #The information for all the epochs
    infomration = np.array(Parallel(n_jobs=NUM_CORES)(delayed(get_information_for_epoch) (i, ws[i], bins, label, b, b1, len(unique_a), p_ys, p_xs, p_y_given_x) for i in range(len(ws))))
    #For only the final epoch
    #infomration = np.array(get_information_for_epoch(-1, ws[-1], bins, label, b, b1, len(unique_a), p_ys, p_xs, p_y_given_x))
    return infomration



def load_data(name):
    """Load the data from mat file
    F is the samples and y is the labels"""
    print ('Loading Data')
    if name =='data/MNIST':
        data_sets  = fetch_mldata('MNIST original')
        labels = np.zeros((len(data_sets.target), 10))
        labels[:, np.array(data_sets.target, dtype=np.int8)] = 1
        data_sets.target = labels
    else :
        d = sio.loadmat(name + '.mat')
        F = d['F']
        y = d['y']
        C = type('type_C', (object,), {})
        data_sets = C()
        data_sets.data =  F
        data_sets.target = np.array(y[0])[:, np.newaxis]
    return data_sets


def batch(iterable, n=1):
    """Return the next batch"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def do_iteration(
        hidden, X_test, labels_test, X_train, labels_train, X_all,
        labels_all, sess, accuracy, optimizer, points,x, y_true):
    """one iteration of the network for all the batchs"""
    for i in xrange(0, len(points) - 1):
        batch_xs = X_train[points[i]:points[i + 1]]
        batch_ys = labels_train[points[i]:points[i + 1]]
        optimizer.run({x: batch_xs, y_true: batch_ys})
    ws = sess.run(hidden, feed_dict={x: X_all, y_true: labels_all})
    test_pred = sess.run(accuracy, feed_dict={x: X_test, y_true: labels_test})
    train_pred = sess.run(accuracy, feed_dict={x: X_train, y_true: labels_train})
    return ws, test_pred, train_pred


def train_network_MNIST(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads,data_sets,sample_size):
    """Train the network on MNIST"""
    input_size = np.array(data_sets.data).shape[1]
    NUM_CLASSES = np.array(data_sets.target).shape[1]
    percent = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        labels = tf.placeholder(tf.int64, shape=[None, NUM_CLASSES], name='y_true')
        hidden, weightsModel = [], []
        with tf.name_scope('hidden0'):
            weights = tf.Variable(
                tf.truncated_normal([input_size, layerSize[0]],
                                    stddev=1.0 / np.sqrt(float(input_size))),
                name='weights')
            biases = tf.Variable(tf.zeros([layerSize[0]]),
                                 name='biases')
            weightsModel.append(weights)
            hidden.append(tf.nn.tanh(tf.matmul(x, weights) + biases))
        for i in xrange(1, len(layerSize)):
            with tf.name_scope('hidden' + str(i)):
                weights = tf.Variable(
                    tf.truncated_normal([layerSize[i - 1], layerSize[i]],
                                        stddev=1.0 / np.sqrt(float(layerSize[i - 1]))),
                    name='weights')
                biases = tf.Variable(tf.zeros([layerSize[i]]),
                                     name='biases')
                weightsModel.append(weights)

                hidden.append(tf.nn.tanh(tf.matmul(hidden[i - 1], weights) + biases))
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([layerSize[-1], NUM_CLASSES],
                                    stddev=1.0 / np.sqrt(float(layerSize[-1]))),
                name='weights')
            biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                                 name='biases')
            weightsModel.append(weights)
            logits = tf.nn.softmax(tf.matmul(hidden[-1], weights) + biases)
        hidden.append(logits)

        labels = tf.to_int64(labels)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        var_grad = tf.gradients(cross_entropy, hidden)
        grads = tf.gradients(cross_entropy, tf.trainable_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_local).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        n_train =percent(sample_size , len(data_sets.data))
        n_test = len(data_sets.data) - n_train
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train + 1, n_train + n_test)
        X_train, y_train = np.array(data_sets.data[train_idx]), np.array(data_sets.target[train_idx])
        X_test, y_test = np.array(data_sets.data[test_idx]), np.array(data_sets.target[test_idx])

        points_of_batch = np.rint(np.arange(0, n_train+1,batch_size)).astype(dtype=np.int32)
        print ('Number of points_of_batch in batch  - ' , len(points_of_batch))
        init = tf.global_variables_initializer()
        ws, ws_test,ws_train,ws_ps,train_pred ,test_pred= [], [],[],[], [],[]
        loss_func_test,loss_func_train,var_grad_val, l1_norm, l2_norm =[],[],[], [],[]
        with tf.Session() as sess:
            sess.run(init)
            #Goes over all the epochs
            for j in range(0, num_of_ephocs):
                epochs_grads = []

                if np.mod(j, 10) ==1:
                    print (j,sess.run(accuracy, feed_dict={x: X_test, labels: y_test}))
                #Goes over all the batches
                for i in xrange(0, len(points_of_batch) - 1):
                    batch_xs = X_train[points_of_batch[i]:points_of_batch[i + 1],:]
                    batch_ys = y_train[points_of_batch[i]:points_of_batch[i + 1],:]
                    optimizer.run({x: batch_xs, labels: batch_ys})
                    epochs_grads.append(sess.run(grads, feed_dict={x: batch_xs, labels: batch_ys}))
                if j in indexes:
                    ws.append(sess.run(hidden, feed_dict={x: X_test, labels:y_test}))
                    loss_func_test.append(sess.run(cross_entropy, feed_dict={x: X_test, labels: y_test}))
                    loss_func_train.append(sess.run(cross_entropy, feed_dict={x: X_train, labels: y_train}))
                    test_pred.append(sess.run(accuracy, feed_dict={x: X_test, labels: y_test}))
                    train_pred.append(sess.run(accuracy, feed_dict={x: X_train, labels: y_train}))
                    if save_grads:
                        var_grad_val.append(epochs_grads)
                    ws_ps.append(sess.run(weightsModel))
                    flatted_list = [sub_item for sublist in sess.run(weightsModel) for item in sublist for sub_item in item]
                    l1_norm.append(LA.norm(np.array(flatted_list), 1))
                    l2_norm.append(LA.norm(np.array(flatted_list)))
    return ws, test_pred, train_pred,  X_test, y_test, loss_func_test, loss_func_train,var_grad_val,l1_norm,l2_norm, flatted_list, ws_ps



def runNetCostum1(layerSize, num_of_ephocs, learning_rate_local, batch_size,indexes,save_grads, data_sets):
    tf.reset_default_graph()
    print ('Load data')
    #data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='x')
        labels = tf.placeholder(tf.int64, shape=[None, NUM_CLASSES], name='y_true')
        hidden, weightsModel = [], []
        with tf.name_scope('hidden0'):
            weights = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS, layerSize[0]],
                                    stddev=1.0 / np.sqrt(float(IMAGE_PIXELS))),
                name='weights')
            biases = tf.Variable(tf.zeros([layerSize[0]]),
                                 name='biases')
            weightsModel.append(weights)
            hidden.append(tf.nn.tanh(tf.matmul(x, weights) + biases))
        for i in xrange(1, len(layerSize)):
            with tf.name_scope('hidden' + str(i)):
                weights = tf.Variable(
                    tf.truncated_normal([layerSize[i - 1], layerSize[i]],
                                        stddev=1.0 / np.sqrt(float(layerSize[i - 1]))),
                    name='weights')
                biases = tf.Variable(tf.zeros([layerSize[i]]),
                                     name='biases')
                weightsModel.append(weights)

                hidden.append(tf.nn.tanh(tf.matmul(hidden[i - 1], weights) + biases))
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([layerSize[-1], NUM_CLASSES],
                                    stddev=1.0 / np.sqrt(float(layerSize[-1]))),
                name='weights')
            biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                                 name='biases')
            weightsModel.append(weights)
            logits = tf.nn.softmax(tf.matmul(hidden[-1], weights) + biases)
        hidden.append(logits)

        labels = tf.to_int64(labels)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        var_grad = tf.gradients(cross_entropy, hidden)
        grads = tf.gradients(cross_entropy, tf.trainable_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_local).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        points = np.rint(np.arange(0, data_sets.train.num_examples+1,batch_size)).astype(dtype=np.int32)
        print ('Number of points in batch  - ' , len(points))
        init = tf.global_variables_initializer()
        train_pred, test_pred = [], []
        ws, ws_test,ws_train,ws_ps = [], [],[],[]
        loss_func_test,loss_func_train,var_grad_val, l1_norm, l2_norm =[],[],[], [],[]

        with tf.Session() as sess:
            sess.run(init)
            for j in range(0, num_of_ephocs):
                epochs_grads = []
                if np.mod(j, 2) ==1:
                    print (j,sess.run(accuracy, feed_dict={x: data_sets.test.images, labels: data_sets.test.labels}))
                for i in xrange(0, len(points) - 1):
                    #batch_xs, batch_ys = data_sets.train.next_batch(batch_size,fake_data)
                    batch_xs = data_sets.train.images[points[i]:points[i + 1],:]
                    batch_ys = data_sets.train.labels[points[i]:points[i + 1],:]
                    optimizer.run({x: batch_xs, labels: batch_ys})
                    epochs_grads.append(sess.run(grads, feed_dict={x: batch_xs, labels: batch_ys}))
                if j in indexes:
                    ws.append(sess.run(hidden, feed_dict={x: data_sets.test.images, labels: data_sets.test.labels}))
                    loss_func_test.append(sess.run(cross_entropy, feed_dict={x: data_sets.test.images, labels: data_sets.test.labels}))
                    loss_func_train.append(sess.run(cross_entropy, feed_dict={x: data_sets.train.images, labels: data_sets.train.labels}))
                    test_pred.append(sess.run(accuracy, feed_dict={x: data_sets.test.images, labels: data_sets.test.labels}))
                    train_pred.append(sess.run(accuracy, feed_dict={x: data_sets.train.images, labels: data_sets.train.labels}))
                    if save_grads:
                        var_grad_val.append(epochs_grads)
                    ws_ps.append(sess.run(weightsModel))
                    flatted_list = [sub_item for sublist in sess.run(weightsModel) for item in sublist for sub_item in item]
                    l1_norm.append(LA.norm(np.array(flatted_list), 1))
                    l2_norm.append(LA.norm(np.array(flatted_list)))
    X_test = data_sets.test.images
    y_test = data_sets.test.labels
    return ws, test_pred, train_pred, X_test, y_test, loss_func_test, loss_func_train,var_grad_val,l1_norm,l2_norm, flatted_list, ws_ps



def train_and_calc_information(calc_information, data_sets, sample_size, layerSize, num_of_ephocs, name_to_save,
                               learning_rate, batch_size, indexes, save_ws, save_grads, num_of_bins):
    data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)
    """Train the network on the given data and calculate the information if nedded"""

    #ws, test_pred, train_error, X_test, X_train,loss_func_test, loss_func_train,var_grad_val,l1_norm,l2_norm,flatted_list,ws_ps  = runNet(F_org, y_org, sampleSize, layerSize, F, y_t, input_size,
    #                                                                        num_of_ephocs, name_to_save,
    #                                                                       learning_rate, batch_size,indexes)
    ws, test_pred, train_error, X_test, y_test, loss_func_test, loss_func_train, var_grad_val, l1_norm, l2_norm, flatted_list, ws_ps = \
        runNetCostum1(layerSize, num_of_ephocs, learning_rate, batch_size, indexes, save_grads,data_sets)
    information_all = get_infomration(ws, X_test,y_test,num_of_bins) if calc_information ==1 else 0
    ws = 0 if not save_ws else ws
    return information_all, test_pred, train_error, loss_func_test, loss_func_train,ws,var_grad_val,l1_norm,l2_norm,flatted_list


def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p].T


def select_network_arch(type_net):
    """Selcet the architectures of the networks according to thier type
    Can we also costume network"""
    if type_net == '1':
        layers_sizes = [[10, 7, 5, 4, 3]]
    elif type_net == '1-2-3':
        layers_sizes = [[10, 9, 7, 7, 3],[10, 9, 7, 5, 3], [10, 9, 7, 3, 3]]
    elif  type_net == '11':
        layers_sizes = [[10, 7, 7, 4, 3]]
    elif type_net == '2':
        layers_sizes = [[10, 7, 5, 4]]
    elif type_net == '3':
        layers_sizes = [[10, 7, 5]]
    elif type_net == '4':
        layers_sizes = [[10, 7]]
    elif type_net == '5':
        layers_sizes = [[10]]
    elif type_net == '6':
        layers_sizes = [[400,200, 150, 60, 50, 40,30]]
    else:
        layers_sizes  = [ map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", type_net) ]
    return layers_sizes


def sample_dist(F, y, num_of_samples):
    """
    For each f in F Samples from the y's distribution num of samples times
    """
    newY = np.empty([y.shape[0], num_of_samples*y.shape[1]], dtype=int)
    newF = np.repeat(F,num_of_samples, axis=0)
    for i in range(y.shape[1]):
        current_samples = np.random.choice([0, 1], size=(num_of_samples,), p=[1-y[0,i],y[0,i]])
        newY[0, i*num_of_samples:(i+1)*num_of_samples ] = current_samples
    return newF, newY

def run_specific_network(
        calc_information, i,j,k , data_sets, samples, layers_sizes, num_of_ephocs,name_to_save,learning_rate,batch_size,
        epochs_indexes,save_ws,save_grads,num_of_bins):
    """Train and calculate the information of on network"""
    print ('Starting network number - ',i,j,k)
    #data_sets.data, data_sets.target = shuffle(data_sets.data, data_sets.target)
    loca_inf_all , local_erorr, local_train,loss_func_test, loss_func_train,ws,var_grad_val,l1_norm,l2_norm,flatted_list_ws= \
                    train_and_calc_information(calc_information,data_sets,  samples, layers_sizes, num_of_ephocs, \
                                               name_to_save, learning_rate, batch_size, epochs_indexes, save_ws, save_grads, num_of_bins)
    return loca_inf_all , local_erorr, local_train,loss_func_test, loss_func_train,ws,var_grad_val,l1_norm,l2_norm,flatted_list_ws


def saveData(data, name_to_save):
    """Save the given data in the given directory"""
    directory = 'jobs/' + name_to_save + '/'
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + 'data.pickle', 'wb') as f:
        cPickle.dump(data, f, protocol=2)
    #Also save the code file
    file_name = os.path.realpath(__file__)
    srcfile = os.getcwd() + '/' +os.path.basename(file_name)

    dstdir = directory + os.path.basename(file_name)
    shutil.copy(srcfile, dstdir)


def train_and_calc_information_all_networks(type_net, name_to_store, new_index, batch_size, learning_rate, num_ephocs, num_of_repeats, data_name,
                                            num_of_samples, num_of_disribuation_samples, start_samples, save_ws, calc_information, save_grads, num_of_bins, run_in_parallel):
    print ("Starting calculations")
    name = 'data/'+ data_name
    layers_sizes = select_network_arch(type_net)
    samples = np.linspace(1, 100, 199)[[[x*2-2 for x in index] for index in new_index]]
    #epochs_indexes = np.arange(0, num_of_ephocs)
    epochs_indexes = np.unique(np.logspace(np.log2(start_samples), np.log2(num_ephocs), num_of_samples, dtype=int, base = 2)) - 1
    max_size = np.max([len(layers_size) for layers_size in layers_sizes])
    data_sets = load_data(name)
    params = {'samples_len' : len(samples), 'num_of_disribuation_samples':num_of_disribuation_samples, 'layersSizes': layers_sizes, 'numEphocs':num_ephocs, 'batch': batch_size,
            'numRepeats': num_of_repeats, 'numEpochsInds': len(epochs_indexes), 'LastEpochsInds': epochs_indexes[-1], 'DataName': data_name, 'learningRate': learning_rate}
    name_to_save = name_to_store+"_"+"_".join([str(i)+'='+str(params[i]) for i in params])
    params['samples'], params['CPUs'], params['nameSave'] , params['directory'] = samples,  NUM_CORES,name_to_store, name_to_save
    for i in params:
        print i, params[i]
    params['epochsInds'] = epochs_indexes
    ws_all = [[[[None] for k in range(len(samples))] for j in range(len(layers_sizes))]for i in range(num_of_repeats)] if save_ws ==1 else 0
    var_grad_val = [[[[None] for k in range(len(samples))] for j in range(len(layers_sizes))]for i in range(num_of_repeats)] if save_grads ==1 else 0
    information = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes), max_size + 1, 2]) if calc_information ==1 else 0
    #information_each_neuron = [[[[None] for k in range(len(samples))] for j in range(len(layers_sizes))]for i in range(num_of_repeats)]
    test_error, train_error = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)]), np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    loss_train, loss_test = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)]), np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    l1_norms, l2_norms = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)]), np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    if run_in_parallel ==1:
        infomration = Parallel(n_jobs=NUM_CORES)(delayed(run_specific_network)
                                      (calc_information, i,j,k , samples[i], layers_sizes[j], num_ephocs,name_to_save,learning_rate,batch_size,epochs_indexes,save_ws,num_of_bins)
                                        for i in range(len(samples)) for j in range(len(layers_sizes)) for k in range(num_of_repeats))
    else:
        infomration= [run_specific_network(
            calc_information, i, j, k, data_sets, samples[i], layers_sizes[j], num_ephocs, name_to_save,
            learning_rate, batch_size, epochs_indexes, save_ws, save_grads, num_of_bins)
                      for i in range(len(samples)) for j in range(len(layers_sizes)) for k in range(num_of_repeats)]
    for i in range(len(samples)):
        for j in range(len(layers_sizes)):
            for k in range(num_of_repeats):
                index= i*len(layers_sizes)*num_of_repeats+j*num_of_repeats+k
                loca_inf_all , test_error[k, j, i, :], train_error[k, j, i, :],loss_test[k, j, i, :], loss_train[k, j, i, :],ws,var_grads , l1_norms[k, j, i, :], \
                l2_norms[k, j, i, :],flatted_list_ws    = infomration[index]
                if calc_information ==1:
                    org_information = loca_inf_all[:, :,:]
                    #per_neuron_information = loca_inf_all[:,1, :,:]
                    current_num_of_layer = org_information.shape[1]
                    information[k, j, i, :, 0:current_num_of_layer, :] = org_information
                    #information_each_neuron[k][j][i] = per_neuron_information
                if save_ws == 1: ws_all[k][j][i] = flatted_list_ws
                if save_grads == 1: var_grad_val[k][j][i] = var_grads
    vars_list = {'information': information, 'test_error': test_error, 'train_error': train_error,
                 'loss_test': loss_test, 'loss_train': loss_train,'params': params, 'var_grad_val': var_grad_val
                    ,'l1_norms': l1_norms,'ws_all': ws_all}
    saveData(vars_list, name_to_save)
    print ('Finished')

def parse_and_run_networks():
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_samples', '-ss', dest="start_samples",default=1, type=int)
    parser.add_argument('-batch_size', '-b', dest="batch_size",default=2544, type=int)
    parser.add_argument('-learning_rate', '-l', dest="learning_rate",default=0.0002, type=float)
    parser.add_argument('-num_repeat', '-r', dest="num_of_repeats",default=1, type=int)
    parser.add_argument('-num_epochs', '-e', dest="num_of_ephocs",default=1400, type=int)
    parser.add_argument('-net', '-n', dest="net_type",default='6')
    parser.add_argument('-inds', '-i', dest="inds",default='[85]')
    parser.add_argument('-name', '-na', dest="name",default='r')
    parser.add_argument('-d_name', '-dna', dest="data_name",default='var_u')
    parser.add_argument('-num_samples', '-ns', dest="num_of_samples",default=1800, type = int)
    parser.add_argument('-num_of_disribuation_samples', '-nds', dest="num_of_disribuation_samples",default=1, type = int)
    parser.add_argument('-save_ws', '-sws', dest="save_ws",default=False)
    parser.add_argument('-calc_information', '-cinf', dest="calc_information",default=1)
    parser.add_argument('-save_grads', '-sgrad', dest="save_grads",default=False)
    parser.add_argument('-run_in_parallel', '-par', dest="run_in_parallel",default=False)
    parser.add_argument('-num_of_bins', '-nbins', dest="num_of_bins",default=50)


    args = parser.parse_args()
    args.inds  =[map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", args.inds)]
    train_and_calc_information_all_networks(args.net_type, args.name, args.inds, args.batch_size, args.learning_rate, args.num_of_ephocs, args.num_of_repeats,
                                            args.data_name, args.num_of_samples, args.num_of_disribuation_samples, args.start_samples, args.save_ws, args.calc_information, args.save_grads,
                                            args.num_of_bins, args.run_in_parallel)

if __name__ == "__main__":
    parse_and_run_networks()
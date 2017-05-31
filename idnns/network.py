"""Train and calculate the infomrmation of network"""
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
import multiprocessing
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
from idnns.information import information_process  as inn
from idnns import model as mo
import os, sys
import  warnings
warnings.filterwarnings("ignore")
summaries_dir = 'summaries'
NUM_CORES = multiprocessing.cpu_count()
def load_data(name, random_labels=False):
    """Load the data
    name - the name of the dataset
    random_labels - True if we want to return random labels to the dataset
    return object with data and labels"""
    print ('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()
    if name.split('/')[-1] =='MNIST':
        data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0])+"/data/MNIST_data/", one_hot=True)
        data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis =0)
        data_sets.labels = np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis =0)
    else :
        d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]),name + '.mat'))
        F = d['F']
        y = d['y']
        C = type('type_C', (object,), {})
        data_sets = C()
        data_sets.data =  F
        data_sets.labels  = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    if random_labels:
        labels = np.zeros(data_sets.labels.shape)
        labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
        labels[np.arange(len(labels)), labels_index] = 1
        data_sets.labels = labels
    return data_sets


def bulid_model(activation_function, layerSize, input_size, num_of_classes, learning_rate_local, save_file, covn_net):
    """Bulid specipic model of the network
    Return the network
    """
    model = mo.Model(input_size, layerSize, num_of_classes, learning_rate_local,save_file, int(activation_function), cov_net = covn_net)
    return model

def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train,min_test_data=80, shuffle_data = False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train>min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels =shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index,:]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:,:]
    return data_sets

def train_and_calc_inf_network(i,j,k,layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads, data_sets_org,
                               model_type,percent_of_train,interval_accuracy_display, calc_information, calc_information_last, num_of_bins,
                               interval_information_display,save_ws, rand_int,cov_net):
    """Train the network and calculate it's information"""
    network_name = '{0}_{1}_{2}_{3}'.format(i,j,k, rand_int)

    print ('Training network  - {0}'.format(network_name))
    network = train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads,
                  data_sets_org, model_type, percent_of_train, interval_accuracy_display, network_name, cov_net)

    if calc_information:
        print ('Calculating the infomration')
        infomration = np.array([inn.get_information(network['ws'], data_sets_org.data, data_sets_org.labels, network['estimted_label'],
                                                    num_of_bins, interval_information_display, indexes, network['model'], layerSize)])
        network['information'] = infomration
    elif calc_information_last:
        print ('Calculating the infomration for the last epoch')
        infomration = np.array([inn.get_information([network['ws'][-1]], data_sets_org.data, data_sets_org.labels,
                                                    network['estimted_label'], num_of_bins, interval_information_display,
                                                    indexes, network['model'], layerSize)])
        network['information'] = infomration
    #If we dont want to save layer's output
    if not save_ws:
        network['ws'] = 0
    return network

def train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads,
                  data_sets_org, model_type,percent_of_train,interval_accuracy_display,
                  name,covn_net):
    """Train the nework"""
    tf.reset_default_graph()
    data_sets = data_shuffle(data_sets_org, percent_of_train)
    ws, estimted_label,gradients, infomration, models= [[None]*len(indexes) for _ in range(5) ]
    loss_func_test, loss_func_train, test_prediction, train_prediction = [np.zeros((len(indexes))) for _ in range(4)]
    input_size = data_sets_org.data.shape[1]
    num_of_classes = data_sets_org.labels.shape[1]
    #The name of the file to store that model
    save_file = 'net_' +name
    batch_size = np.min([batch_size,data_sets.train.data.shape[0]])
    batch_points = np.rint(np.arange(0, data_sets.train.data.shape[0]+1,batch_size)).astype(dtype=np.int32)
    batch_points_test = np.rint(np.arange(0, data_sets_org.data.shape[0]+1,batch_size)).astype(dtype=np.int32)
    if data_sets_org.data.shape[0] not in batch_points_test:
        batch_points_test = np.append(batch_points_test, [data_sets_org.data.shape[0]])
    if data_sets.train.data.shape[0] not in batch_points:
        batch_points = np.append(batch_points, [data_sets.train.data.shape[0]])
    #Build the network
    model = bulid_model(model_type,layerSize, input_size, num_of_classes,learning_rate_local, save_file,covn_net)
    optimizer = model.optimize
    saver = tf.train.Saver(max_to_keep=0)
    init = tf.global_variables_initializer()
    grads = tf.gradients(model.cross_entropy, tf.trainable_variables())
    feed_dict_all =  {model.x: data_sets_org.data, model.labels: data_sets_org.labels}
    feed_dict_train =  {model.x: data_sets.train.data, model.labels: data_sets.train.labels}
    feed_dict_test =  {model.x: data_sets.test.data, model.labels: data_sets.test.labels}

    if covn_net:
        feed_dict_test[model.drouput] = 1
        feed_dict_all[model.drouput] = 1
        feed_dict_train[model.drouput] = 1
    #Train the network
    with tf.Session() as sess:
        sess.run(init)
        #Go over the epochs
        k =0
        for j in range(0, num_of_ephocs):
            epochs_grads = []
            if j in indexes:
                ws[k], estimted_label[k] = \
                sess.run([model.hidden_layers, model.prediction],
                             feed_dict=feed_dict_all)
                loss_func_train[k], train_prediction[k] = \
                    sess.run([ model.cross_entropy, model.accuracy],
                             feed_dict=feed_dict_train)

                loss_func_test[k], test_prediction[k] = \
                    sess.run([model.cross_entropy, model.accuracy],
                             feed_dict=feed_dict_test)
                """"
                infomration[k] = inn.calc_information_for_epoch(k, interval_information_display, ws_t, params['bins'],
                                                  params['unique_inverse_x'],
                                                  params['unique_inverse_y'],
                                                  params['label'], estimted_labels,
                                                  params['b'], params['b1'], params['len_unique_a'],
                                                  params['pys'], py_hats_temp, params['pxs'], params['py_x'],
                                                  params['pys1'])

                """
            # Print accuracy every some epochs
            if np.mod(j, interval_accuracy_display) ==1 or interval_accuracy_display ==1:
                print ('Epoch {0} - Test Accuracy: {1:.3f}, Train Accuracy: {2:.3f}'.format(j,sess.run(model.accuracy, feed_dict=feed_dict_test),
                                                                                            sess.run(model.accuracy,
                                                                                                     feed_dict=feed_dict_train)))
            #Go over the batch_points
            for i in xrange(0, len(batch_points) - 1):
                batch_xs = data_sets.train.data[batch_points[i]:batch_points[i + 1]]
                batch_ys = data_sets.train.labels[batch_points[i]:batch_points[i + 1]]
                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                if covn_net:
                    feed_dict[model.drouput] = 0.5
                optimizer.run(feed_dict)
                if j in indexes:
                    epochs_grads_temp, loss_tr, tr_err = sess.run([grads, model.cross_entropy, model.accuracy],
                                                             feed_dict=feed_dict)
                    epochs_grads.append(epochs_grads_temp)
            if j in indexes:
                if save_grads:
                    gradients[k] = epochs_grads
                #Save the model
                write_meta = True if k == 0 else False
                saver.save(sess, model.save_file, global_step=k, write_meta_graph=write_meta)
                k+=1
    network = {}
    network['ws'] = ws
    network['estimted_label'] = estimted_label
    network['test_prediction'] = test_prediction
    network['train_prediction'] = train_prediction
    network['loss_test'] = loss_func_test
    network['loss_train'] = loss_func_train
    network['gradients'] = gradients
    network['model'] = model
    return network


import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
import multiprocessing
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
from idnns.information import information_process  as inn
from idnns import model as mo
summaries_dir = 'summaries'
NUM_CORES = multiprocessing.cpu_count()

def load_data(name, random_labels):
    """Load the data from mat file
    F is the samples and y is the labels"""
    print ('Loading Data')
    #print name.split('/')[-1]
    print name
    if name.split('/')[-1] =='MNIST':
        """"
        data_sets  = fetch_mldata('MNIST original')
        labels = np.zeros((len(data_sets.target), 10))
        labels[:, np.array(data_sets.target, dtype=np.int8)] = 1
        data_sets.target = labels
        """
        if False:
            pre = '/Users/ravidziv/PycharmProjects/IDNNs/'
        else:
            pre = ''
        data_sets_temp = input_data.read_data_sets(pre+"MNIST_data/", one_hot=True)
        C = type('type_C', (object,), {})
        data_sets = C()
        data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis =0)
        data_sets.labels = np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis =0)

    else :
        import os
        print os.path.dirname(os.path.abspath(__file__))

        d = sio.loadmat(name + '.mat')
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

def bulid_model(model_type,layerSize, input_size, num_of_classes,learning_rate_local, save_file,covn_net):
    """Bulid specipic model of the network
    model_type ==2 is for relu"""
    model = mo.Model(input_size, layerSize, num_of_classes,learning_rate_local, '/Users/ravidziv/PycharmProjects/IDNNs/'
                     +'net_logs/' +save_file,int(model_type),cov_net = covn_net)
    return model

def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train):
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train>80:
        start_test_index = perc(80, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    #shuffled_data, shuffled_labels =shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index,:]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:,:]
    return data_sets

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

def train_and_calc_inf_network(i,j,k,layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads, data_sets_org,
                               model_type,percent_of_train,interval_accuracy_display, calc_information, calc_information_last, num_of_bins,
                               interval_information_display,save_ws, rand_int,cov_net):
    print ('Training network number {0},{1},{2}'.format(i,j,k))
    name = '{0}_{1}_{2}_{3}'.format(i,j,k, rand_int)
    if int(cov_net) == 0 :
        cov_net = False

    ws, estimted_label, test_pred, train_pred, loss_func_test, loss_func_train, grads , model,infomration = \
        train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads,
                  data_sets_org, model_type, percent_of_train, interval_accuracy_display, name,interval_information_display,num_of_bins, cov_net)
    #infomration = 0
    if False and calc_information =='1':
        print ('Calculating the infomration')
        infomration = np.array([inn.get_infomration(ws, data_sets_org.data, data_sets_org.labels, estimted_label,
                                      num_of_bins, interval_information_display, indexes, model)])

    elif False and calc_information_last == '1':
        print ('Calculating the infomration last')
        infomration = np.array([inn.get_infomration([ws[-1]], data_sets_org.data, data_sets_org.labels,
                                                    num_of_bins, interval_information_display, indexes)])
    #If we dont want to save everything
    if save_ws !='1':
        ws = 0
    return infomration, ws, test_pred, train_pred, loss_func_test, loss_func_train, model, name, grads
def get_inf_params(x, label,  num_of_bins):
    label = np.array(label).astype(np.float)
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pys = np.sum(label, axis=0) / float(label.shape[0])
    # pys = [prob_y, 1 - prob_y]

    # prob_y_hat = [np.sum(estimted_label_local) / float(estimted_label_local.shape[1]) for estimted_label_local in estimted_label]
    # py_hats = [[local_p_y_hat, 1 - local_p_y_hat] for local_p_y_hat in prob_y_hat]
    py_hats = 0
    pxs = unique_counts / float(np.sum(unique_counts))
    bins = np.linspace(-1, 1, num_of_bins)
    py_x = []
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        py_x.append(py_x_current)
    py_x = np.array(py_x).T
    py_x = np.array([py_x, 1 - py_x])
    print ('Starting Calculating The information...')
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    params ={}
    params['bins'] = bins
    params['pys1'] = pys1
    params['unique_inverse_x'] = unique_inverse_x
    params['unique_inverse_y'] = unique_inverse_y
    params['label'] = label
    params['b'] = b
    params['b1'] = b1
    params['len_unique_a'] = len(unique_a)
    params['pys']= pys
    params['pxs'] = pxs
    params['py_x'] = py_x
    return params


def train_network(layerSize, num_of_ephocs, learning_rate_local, batch_size, indexes, save_grads,
                  data_sets_org, model_type,percent_of_train,interval_accuracy_display,
                  name,interval_information_display,num_of_bins,covn_net):
    """Train the nework"""
    save_grads = False if save_grads ==0 else True
    data_sets = data_shuffle(data_sets_org, percent_of_train)
    #Bulid the network all the hidden layers are tanh and the final layer is soft-max
    ws, estimted_label,gradients, infomration, models= [[None]*len(indexes) for _ in range(5) ]
    loss_func_test, loss_func_train, test_prediction, train_prediction = [np.zeros((len(indexes))) for _ in range(4)]
    tf.reset_default_graph()
    input_size = data_sets_org.data.shape[1]
    num_of_classes = data_sets_org.labels.shape[1]
    save_file = 'net_' +name
    #covn_net = True
    model = bulid_model(model_type,layerSize, input_size, num_of_classes,learning_rate_local, save_file,covn_net)
    #The loss function
    batch_size = np.min([batch_size,data_sets.train.data.shape[0]])
    batch_points = np.rint(np.arange(0, data_sets.train.data.shape[0]+1,batch_size)).astype(dtype=np.int32)
    batch_points_test = np.rint(np.arange(0, data_sets_org.data.shape[0]+1,batch_size)).astype(dtype=np.int32)
    if data_sets_org.data.shape[0] not in batch_points_test:
        batch_points_test = np.append(batch_points_test, [data_sets_org.data.shape[0]])

    if data_sets.train.data.shape[0] not in batch_points:
        batch_points = np.append(batch_points, [data_sets.train.data.shape[0]])
    optimizer = model.optimize
    #ass = [tf.assign(v, 1) for v in model.hidden_layers]
    #dict = {v.op.name: v for v in ass}
    saver = tf.train.Saver(max_to_keep=0)
    init = tf.global_variables_initializer()
    #Train the network
    params = get_inf_params(data_sets_org.data, data_sets_org.labels,  num_of_bins)
    with tf.Session() as sess:
        grads = tf.gradients(model.cross_entropy, tf.trainable_variables())

        sess.run(init)
        #Go over the epochs
        k =0
        print len(batch_points)
        for j in range(0, num_of_ephocs):
            epochs_grads = []
            #Print accuracy every some epochs
            if j in indexes:

                test_prediction_temp_arr, estimted_label_arr, loss_func_test_temp_arr, ws_temp_array = [], [], [], []
                for i in xrange(0, len(batch_points_test) - 1):
                    batch_xs = data_sets_org.data[batch_points_test[i]:batch_points_test[i + 1]]
                    batch_ys = data_sets_org.labels[batch_points_test[i]:batch_points_test[i + 1]]
                    feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                    if covn_net:
                        feed_dict = {model.drouput: 1, model.x: batch_xs, model.labels: batch_ys}

                    ws_temp, estimted_label_temp, loss_func_test_temp, test_prediction_temp = \
                        sess.run([model.hidden_layers, model.prediction, model.cross_entropy, model.accuracy],
                                 feed_dict=feed_dict)
                    ws_temp_array.append(ws_temp)
                    estimted_label_arr.append(estimted_label_temp)
                    test_prediction_temp_arr.append(np.array(test_prediction_temp))
                    loss_func_test_temp_arr.append(np.array(loss_func_test_temp))


                # ws[k] = np.array(ws_temp_array)
                ws_t = []
                for batch_ws in ws_temp_array:
                    for layze_index, layer_ws in enumerate(batch_ws):
                        if len(ws_t) <= layze_index:
                            ws_t.append(layer_ws)
                        else:
                            ws_t[layze_index] = np.concatenate((ws_t[layze_index], layer_ws))
                # ws[k] = ws_t
                estimted_label[k] = np.array(estimted_label_arr)
                loss_func_test[k], test_prediction[k] = np.mean(loss_func_test_temp_arr), np.mean(test_prediction_temp_arr)
                #print ('test', test_prediction[k])
                py_hats_temp = 0
                estimted_labels = 0
                ws[k] = ws_t

                #models.append(current_model)
                # print ('before')

                """"
                infomration[k] = inn.processInput(k, interval_information_display, ws_t, params['bins'],
                                                  params['unique_inverse_x'],
                                                  params['unique_inverse_y'],
                                                  params['label'], estimted_labels,
                                                  params['b'], params['b1'], params['len_unique_a'],
                                                  params['pys'], py_hats_temp, params['pxs'], params['py_x'],
                                                  params['pys1'])

                """
                #print infomration[k]
            # print ('after')
            if np.mod(j, int(interval_accuracy_display)) ==1 or int(interval_accuracy_display)==1:
                batch_xs = data_sets_org.data
                batch_ys = data_sets_org.labels
                feed_dict_test = {model.x: batch_xs, model.labels: batch_ys}
                if covn_net:
                    feed_dict_test = {model.drouput:1, model.x: batch_xs, model.labels: batch_ys}
                print ('The accuracy at epoch {0} -  {1:.3f}'.format(j,sess.run(model.accuracy, feed_dict=feed_dict_test)))

            #Go over the batch batch_points
            train_err = []
            loss_train_err = []
            for i in xrange(0, len(batch_points) - 1):
                #print i
                batch_xs = data_sets.train.data[batch_points[i]:batch_points[i + 1]]
                batch_ys = data_sets.train.labels[batch_points[i]:batch_points[i + 1]]

                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                if covn_net:
                    feed_dict =  {model.drouput:0.5, model.x: batch_xs, model.labels: batch_ys}
                optimizer.run(feed_dict)
                #epochs_grads.append(sess.run(grads, feed_dict=feed_dict))
                if j in indexes:
                    epochs_grads_temp, loss_tr, tr_err = sess.run([grads, model.cross_entropy, model.accuracy],
                                                             feed_dict=feed_dict)

                    loss_train_err.append(loss_tr)
                    train_err.append(tr_err)
                    epochs_grads.append(epochs_grads_temp)
            if j in indexes:
                batch_xs = data_sets.train.data
                batch_ys = data_sets.train.labels
                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                if covn_net:
                    feed_dict = {model.drouput: 1, model.x: batch_xs, model.labels: batch_ys}
                epochs_grads_temp, loss_tr, tr_err = sess.run([grads, model.cross_entropy, model.accuracy],
                                                              feed_dict=feed_dict)
                #loss_func_train[k] = np.mean(loss_train_err)
                #train_prediction = np.mean(train_err)
                train_prediction = tr_err
                loss_func_train[k] = loss_tr
                print ('train', train_prediction)
            #Calculate all the measures if we are in emodelspoch to insert
            if j in indexes:
                gradients[k] = epochs_grads

                if not save_grads:
                    gradients[k] =None
                write_meta = True if k == 0 else False
                saver.save(sess, model.save_file, global_step=k, write_meta_graph=write_meta)
                k+=1
    infomration = 0
    if False:
        infomration = np.array(Parallel(n_jobs=NUM_CORES
                                            )(delayed(inn.processInput)(i, interval_information_display, ws[i], params['bins'],
                                         params['unique_inverse_x'],
                                         params['unique_inverse_y'],
                                         params['label'], estimted_labels,
                                         params['b'], params['b1'], params['len_unique_a'],
                                         params['pys'], py_hats_temp, params['pxs'], params['py_x'],
                                         params['pys1'], model.save_file,input_size, layerSize)
                                              for i in range(len(ws))))

    elif True:
        infomration = [inn.processInput(i, interval_information_display, ws[i], params['bins'],
                                         params['unique_inverse_x'],
                                         params['unique_inverse_y'],
                                         params['label'], estimted_labels,
                                         params['b'], params['b1'], params['len_unique_a'],
                                         params['pys'], py_hats_temp, params['pxs'], params['py_x'],
                                         params['pys1'], model.save_file,input_size, layerSize)
                                              for i in range(len(ws))]



    return ws, estimted_label, test_prediction, train_prediction, loss_func_test, loss_func_train, gradients, model,np.array(infomration)


import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cPickle
import matplotlib.pyplot as plt
import time
import shutil
import os
from joblib import Parallel, delayed
import multiprocessing
import sys
import re
total_index_i = 0
import argparse
import re
NUM_CORES = multiprocessing.cpu_count()
from scipy.optimize import fmin, fminbound
def KL(a, b):
    D_KL = np.nansum(np.multiply(a, np.log(np.divide(a, b+np.spacing(1)))))
    return D_KL

def caclIB(p_t_x, PXs, PYgivenXs, beta):
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

def mseFunc(beta, p_t_x, PXs, PYgivenXs):
    probTgivenXs, PYgivenTs_update = caclIB(p_t_x, PXs, PYgivenXs, beta)
    D_KL = np.mean([KL(p_x_given_t_new, px_given_t) for p_x_given_t_new, px_given_t in zip(probTgivenXs.T, p_t_x.T)])
    return D_KL


def calcXI(probTgivenXs,PYgivenTs, PXs, PYs):
    #probTgivenXs = probTgivenXs.astype(np.longdouble)
    #PYgivenTs = PYgivenTs.astype(np.longdouble)
    #PXs =  PXs.astype(np.longdouble)
    #PYs = PYs.astype(np.longdouble)
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = np.sum([np.nansum(np.log2(prob_y_given_t /PYs) * (prob_y_given_t *prob_t))
                  for prob_y_given_t, prob_t in zip(PYgivenTs.T, PTs)], axis=0)

    IYT = Hy - Hyt
    ITX = Ht - Htx
    #print (Hyt)
    return ITX, IYT



def caclInfomration(PXgivenTs, PYgivenTs, PTs, PYs, PXs, PYgivenXs, old_beta):
    PXgivenTs = np.vstack(PXgivenTs).T
    PYgivenTs = np.vstack(PYgivenTs).T
    #PYgivenTs = PYgivenTs.astype(np.longdouble)
    PXs = np.asarray(PXs).T
    PYs = np.asarray(PYs).T
    #PYs = PYs.astype(np.longdouble)
    PYgivenXs =  np.vstack(PYgivenXs).T
    PYgivenXs = np.row_stack((PYgivenXs,1-PYgivenXs))
    #PYgivenXs = PYgivenXs.astype(np.longdouble)
    PTs = np.asarray(PTs, dtype=np.float64).T
    PTgivenXs_not_divide =np.multiply(PXgivenTs,np.tile(PTs,(PXgivenTs.shape[0],1))).T
    PTgivenXs = np.multiply(PTgivenXs_not_divide, np.tile((1./(PXs)), (PXgivenTs.shape[1], 1)))
    ITX, IYT = calcXI(PTgivenXs,PYgivenTs, PXs, PYs)
    return ITX, IYT


def processInputProb(t_index, unique_inverse, label, b, b1, len_unique_a):
    indexs = unique_inverse == t_index
    p_y = np.sum(label[0][indexs]) / len(label[0][indexs])
    p_y_ts = np.array([p_y, 1 - p_y])
    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
    return p_x_ts, p_y_ts


def processInputIter(data, bins, unique_inverse, label, b, b1, len_unique_a, pys, pxs, py_x, beta):
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    pxys = np.array(
        [processInputProb(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
    )
    local_IXT, local_ITY = caclInfomration(pxys[:, 0], pxys[:, 1], p_ts, pys, pxs, py_x, beta)
    return local_IXT, local_ITY


def processInput(iter_index, ws_iter_index, bins, unique_inverse, label, b, b1, len_unique_a, pys, pxs, py_x):
    betas = [20, 30, 40, 60, 100, 100, 100]
    betas.reverse()
    iter_infomration = np.array(
        [processInputIter(ws_iter_index[i], bins, unique_inverse, label, b, b1, len_unique_a, pys, pxs, py_x,
                          betas[i]) for i in range(0, len(ws_iter_index))])
    return iter_infomration

def getProb(ws, x, label):
    label = np.array(label).astype(np.float)
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    prob_y = np.sum(label) / float(label.shape[1])
    pys = [prob_y, 1 - prob_y]
    pxs = unique_counts / float(sum(unique_counts))
    bins = np.linspace(-1, 1, 50)
    py_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse == i
        py_x_current = np.sum(label[0][indexs]) / len(label[0][indexs])
        py_x.append(py_x_current)
    infomration = np.array(Parallel(n_jobs=NUM_CORES)(delayed(processInput)
                                                      (i,ws[i], bins,unique_inverse,label, b, b1,len(unique_a), pys, pxs,py_x)
                                           for i in range(len(ws))))
    #infomration = np.array(
    #    processInput(1, ws[0], bins, unique_inverse, label, b, b1, len(unique_a), pys, pxs, py_x))
    return infomration


def loadData(name):
    d = sio.loadmat(name + '.mat')
    F = d['F']
    y = d['y']
    return F, y


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def trainNets(hidden, X_test, labels_test, X_train, labels_train, X_all, labels_all, sess, accuracy, optimizer, points,
              x, y_true):
    for i in xrange(0, len(points) - 1):
        batch_xs = X_train[points[i]:points[i + 1]]
        batch_ys = labels_train[points[i]:points[i + 1]]
        optimizer.run({x: batch_xs, y_true: batch_ys})
    ws = sess.run(hidden, feed_dict={x: X_all, y_true: labels_all})
    test_pred = sess.run(accuracy, feed_dict={x: X_test, y_true: labels_test})
    train_pred = sess.run(accuracy, feed_dict={x: X_train, y_true: labels_train})
    return ws, test_pred, train_pred


def runNet(F_org, y_org, sampleSize, layerSize, F, y_t, input_size, num_of_ephocs, name_file, learning_rate_local, batch_size,indexes):
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
        hidden = []
        with tf.name_scope('hidden0'):
            weights = tf.Variable(
                tf.truncated_normal([input_size, layerSize[0]],
                                    stddev=1.0 / np.sqrt(float(input_size))),
                name='weights')
            biases = tf.Variable(tf.zeros([layerSize[0]]),
                                 name='biases')
            hidden.append(tf.nn.tanh(tf.matmul(x, weights) + biases))
        for i in xrange(1, len(layerSize)):
            with tf.name_scope('hidden' + str(i)):
                weights = tf.Variable(
                    tf.truncated_normal([layerSize[i - 1], layerSize[i]],
                                        stddev=1.0 / np.sqrt(float(layerSize[i - 1]))),
                    name='weights')
                biases = tf.Variable(tf.zeros([layerSize[i]]),
                                     name='biases')
                hidden.append(tf.nn.tanh(tf.matmul(hidden[i - 1], weights) + biases))
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([layerSize[-1], 2],
                                    stddev=1.0 / np.sqrt(float(layerSize[-1]))),
                name='weights')
            biases = tf.Variable(tf.zeros([2]),
                                 name='biases')
            logits = tf.nn.softmax(tf.matmul(hidden[-1], weights) + biases)
        hidden.append(logits)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(logits), reduction_indices=[1]))
        #learning_rate=learning_rate_local
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_local).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        y_t_trans = np.squeeze(np.concatenate((y_t[None, :], 1 - y_t[None, :]), axis=0).T)
        
        y_org_trans = np.squeeze(np.concatenate((y_org[None, :], 1 - y_org[None, :]), axis=0).T)

        X_test = F[perc(80, y_t_trans.shape[0]):, :]
        labels_test = y_t_trans[perc(80, y_t_trans.shape[0]):, :]
        X_train = F[0:perc(sampleSize, y_t_trans.shape[0]), :]
        labels_train = y_t_trans[0:perc(sampleSize, y_t_trans.shape[0])]
        
        #points = np.rint(np.arange(0, len(X_train)+1,batch_size)).astype(dtype=np.int32)
        points = [0, len(X_train) - 1]
        init = tf.global_variables_initializer()
        train_pred, test_pred = [], []
        ws, ws_test,ws_train = [], [],[]
        loss_func_test,loss_func_train =[],[]
        with tf.Session() as sess:
            sess.run(init)
            for j in range(0, num_of_ephocs):
                if np.mod(j, 4499) ==1:
                    print (j,sess.run(accuracy, feed_dict={x: X_test, y_true: labels_test}))
                for i in xrange(0, len(points) - 1):
                    batch_xs = X_train[points[i]:points[i + 1]]
                    batch_ys = labels_train[points[i]:points[i + 1]]
                    optimizer.run({x: batch_xs, y_true: batch_ys})
                if j in indexes:                
                    ws.append(sess.run(hidden, feed_dict={x: F_org, y_true: y_org_trans}))
                    loss_func_test.append(sess.run(cross_entropy, feed_dict={x: X_test, y_true: labels_test}))
                    loss_func_train.append(sess.run(cross_entropy, feed_dict={x: X_train, y_true: labels_train}))                   
                    test_pred.append(sess.run(accuracy, feed_dict={x: X_test, y_true: labels_test}))
                    train_pred.append(sess.run(accuracy, feed_dict={x: X_train, y_true: labels_train}))
    return ws, test_pred, train_pred, X_test, X_train, loss_func_test, loss_func_train


def calcAndRun(F_org, y_org, sampleSize, layerSize, F, y_t, input_size, num_of_ephocs, name_to_save, 
        learning_rate, batch_size,indexes,save_ws):
    t = time.time()
    #print (y_squeeze)
    #global total_index_i
    #print ('Starting Training network - ' + str(total_index_i))
    #total_index_i += 1
    ws, test_pred, train_error, X_test, X_train,loss_func_test, loss_func_train  = runNet(F_org, y_org, sampleSize, layerSize, F, y_t, input_size,
                                                                            num_of_ephocs, name_to_save,
                                                                            learning_rate, batch_size,indexes)
    elapsed = time.time() - t
    #print ('Statrting Calculation of information - ', str(elapsed))
    t = time.time()
    information_all = getProb(ws, F_org, y_org)
    elapsed = time.time() - t
    #print ('Finished Calculation of information - ', str(elapsed))
    if not save_ws:
        ws = 0
    return information_all, test_pred, train_error, loss_func_test, loss_func_train,ws

    
def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p].T

def chooseNet(type_net):
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
    else:
        print 'costum network!!!!'
        layers_sizes  = [ map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", type_net) ]
    return layers_sizes
"""
For each f in F Samples from the y's distribution num of samples times
"""
def sampleDist(F,y, num_of_samples):
    newY = np.empty([y.shape[0], num_of_samples*y.shape[1]], dtype=int)
    newF = np.repeat(F,num_of_samples, axis=0)
    for i in range(y.shape[1]):
        current_samples = np.random.choice([0, 1], size=(num_of_samples,), p=[1-y[0,i],y[0,i]])
        newY[0, i*num_of_samples:(i+1)*num_of_samples ] = current_samples
    #print ()
    return newF, newY    

def runNetParallel(i,j,k , F,y, samples, layers_sizes,input_size, num_of_ephocs,name_to_save,learning_rate,batch_size,
        epochs_indexes,save_ws):
    print ('Starting network number - ',i,j,k)
    random_F, random_y = shuffle_in_unison_inplace(F, y.T)
    #return [i], [j,j], [k,k,k]
    loca_inf_all , local_erorr, local_train,loss_func_test, loss_func_train,ws = \
                    calcAndRun(F, y, samples,layers_sizes,random_F, random_y,input_size,num_of_ephocs,\
                    name_to_save,learning_rate,batch_size,epochs_indexes,save_ws) 
    return loca_inf_all , local_erorr, local_train,loss_func_test, loss_func_train,ws
def main(type_net, name_to_store, new_index, batch_size,learning_rate,num_of_ephocs,num_of_repeats,data_name,
        num_of_samples,num_of_disribuation_samples,start_samples,save_ws):
    print ("Starting calculation")
    name = 'data/' +data_name
    new_index_string= re.sub('[(){}<>]', '', str(new_index))
    layers_sizes = chooseNet(type_net)
    samples = np.linspace(1, 100, 199)[new_index]
    #epochs_indexes = np.arange(0, num_of_ephocs)
    epochs_indexes = np.unique(np.logspace(np.log2(start_samples), np.log2(num_of_ephocs), num_of_samples, dtype=int, base = 2))-1
    max_size = np.max([len(layers_size) for layers_size in layers_sizes])
    [F, y] = loadData(name)
    input_size = len(F[0])
    params = {}   
    params['samples'] = len(samples)
    params['num_of_disribuation_samples'] = num_of_disribuation_samples
    params['layersSizes'] = layers_sizes
    params['numEphocs'] = num_of_ephocs
    params['numRepeats'] = num_of_repeats
    params['numEpochsInds'] = len(epochs_indexes)
    params['LastEpochsInds'] = epochs_indexes[-1]
    params['DataName'] = data_name
    params['learningRate'] = learning_rate
    params['batch'] = batch_size
    name_to_save = name_to_store+"_"+"_".join([str(i)+'='+str(params[i]) for i in params])
    directory = 'jobs/'+name_to_save+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)   
    params['samples'] = samples

    params['CPUs'] = NUM_CORES
    params['name']=name_to_store
    params['nameSave'] = name_to_save
    for i in params:
        print i, params[i] 
    params['epochsInds'] = epochs_indexes
    information_all = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes), max_size + 1, 2])
    if save_ws ==1:
        print ('TTTTTTTTTTTTT')
        ws_all = [[[[None] for k in range(len(samples))] for j in range(len(layers_sizes))]for i in range(num_of_repeats)]
    else :
        ws_all = 0
    #print (len(ws_all), len(ws_all[0]),len(ws_all[0][0]) )
    test_error = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    train_error = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    loss_train = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    loss_test = np.zeros([num_of_repeats, len(layers_sizes), len(samples), len(epochs_indexes)])
    #F_new, y_new = sampleDist(F,y, num_of_disribuation_samples)
    infomration = Parallel(n_jobs=NUM_CORES)(delayed(runNetParallel)
                                      (i,j,k ,F,y, samples[i], layers_sizes[j],input_size, num_of_ephocs,name_to_save,learning_rate,batch_size,epochs_indexes,save_ws)
                                        for i in range(len(samples)) for j in range(len(layers_sizes)) for k in range(num_of_repeats))
    print len(infomration), len(infomration[0]), len(infomration[0][1]) 
    for i in range(len(samples)):
        for j in range(len(layers_sizes)):
            for k in range(num_of_repeats):            
                index= i*len(layers_sizes)*num_of_repeats+j*num_of_repeats+k                
                loca_inf_all , local_erorr, local_train,loss_func_test, loss_func_train,ws    = infomration[index] 
                current_num_of_layer = loca_inf_all.shape[1]
                information_all[k, j, i, :, 0:current_num_of_layer, :] = loca_inf_all
                test_error[k, j, i, :] = local_erorr
                train_error[k, j, i, :] = local_train
                loss_train[k, j, i, :] = loss_func_train
                loss_test[k, j, i, :] = loss_func_test
                if save_ws ==1:
                    ws_all[k][j][i] = ws
                
    data = {}
    data['information'] = information_all
    data['test_error'] = test_error
    data['train_error'] = train_error
    data['loss_train'] = loss_train
    data['loss_test'] = loss_test
    
    data['ws_all'] = ws_all

    data['params'] = params
    with open(directory + 'data.pickle', 'wb') as f:
        cPickle.dump(data, f, protocol=2)
    srcfile = 'source/main_fas.py'
    dstdir = directory+'main_fas.py'
    shutil.copy(srcfile, dstdir)
    print ('Finished')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_samples', '-ss', dest="start_samples",default=1, type=int)
    parser.add_argument('-batch_size', '-b', dest="batch_size",default=3563, type=int)
    parser.add_argument('-learning_rate', '-l', dest="learning_rate",default=0.0004, type=float)
    parser.add_argument('-num_repeat', '-r', dest="num_of_repeats",default=40, type=int)
    parser.add_argument('-num_epochs', '-e', dest="num_of_ephocs",default=1000, type=int)
    parser.add_argument('-net', '-n', dest="net_type",default='1')
    parser.add_argument('-inds', '-i', dest="inds",default='[198]')
    parser.add_argument('-name', '-na', dest="name",default='r')
    parser.add_argument('-d_name', '-dna', dest="data_name",default='var_u')
    parser.add_argument('-num_samples', '-ns', dest="num_of_samples",default=1800, type = int)
    parser.add_argument('-num_of_disribuation_samples', '-nds', dest="num_of_disribuation_samples",default=1, type = int)
    parser.add_argument('-save_ws', '-sws', dest="save_ws",default=True)

    args = parser.parse_args()
    #args.inds = [ args.inds]
    args.inds  =[map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", args.inds)]
    main(args.net_type, args.name, args.inds, args.batch_size, args.learning_rate, args.num_of_ephocs, args.num_of_repeats,
    args.data_name, args.num_of_samples,args.num_of_disribuation_samples, args.start_samples, args.save_ws )
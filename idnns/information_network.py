import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import network as nn
import os
import cPickle
import shutil
import re
import argparse
from idnns.information import information_process  as inn
import  tensorflow as tf
NUM_CORES = multiprocessing.cpu_count()

def get_default_parser(num_of_samples = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_samples',
                        '-ss', dest="start_samples", default=1,
                        type=int,help='The number of the first sample that we calculate the information')

    parser.add_argument('-batch_size',
                        '-b', dest="batch_size", default=4016,
                        type=int, help='The size of the batch')

    parser.add_argument('-learning_rate',
                        '-l', dest="learning_rate", default=0.0004,
                        type=float,
                        help='The learning rate of the network')

    parser.add_argument('-num_repeat',
                        '-r', dest="num_of_repeats", default=40,
                        type=int,help='The number of times to run the network')

    parser.add_argument('-num_epochs',
                        '-e', dest="num_ephocs", default=10000,
                        type=int, help='max number of epochs')

    parser.add_argument('-net',
                        '-n', dest="net_type", default='1',
                        help='The architecture of the networks')

    parser.add_argument('-inds',
                        '-i', dest="inds", default='[100]',
                        help='The percent of the training data')

    parser.add_argument('-name',
                        '-na', dest="name", default='ttte',
                        help='The name to save the results')

    parser.add_argument('-d_name',
                        '-dna', dest="data_name", default='var_u',
                        help='The dataset n')

    parser.add_argument('-num_samples',
                        '-ns', dest="num_of_samples", default=1800,
                        type=int,
                        help='The max number of indexes for calculate information')

    parser.add_argument('-num_of_disribuation_samples',
                        '-nds', dest="num_of_disribuation_samples", default=1,
                        type=int,help='S')

    parser.add_argument('-save_ws',
                        '-sws', dest="save_ws", default=False,
                        help='if we want to save the weights of the network')

    parser.add_argument('-calc_information',
                        '-cinf', dest="calc_information", default=False,
                        help='1 if we want to calculate the MI in the network')

    parser.add_argument('-calc_information_last',
                        '-cinfl', dest="calc_information_last", default=False,
                        help='1 if we want to calculate the MI in the network')

    parser.add_argument('-save_grads',
                        '-sgrad', dest="save_grads", default=False,
                        help='Save tge gradients in the network')

    parser.add_argument('-run_in_parallel',
                        '-par', dest="run_in_parallel", default=False,
                        help='If we want to run all the networks in parallel mode')

    parser.add_argument('-num_of_bins',
                        '-nbins', dest="num_of_bins", default=50,
                        help='The number of bins that we divide the neurons output')

    parser.add_argument('-calc_vectors',
                        '-cvec', dest="calc_vectors", default=False,
                        help='if true calculate the eigenvectores of the matrices')

    parser.add_argument('-model_type',
                        '-mtype', dest="model_type", default=0,
                        help='ho to bulid the network')

    parser.add_argument('-iad',dest="interval_accuracy_display", default=4990,
                        help='The interval for dipslay accuracy')

    parser.add_argument('-interval_information_display',
                        '-iid', dest="interval_information_display", default=100,
                        help='The interval for dipslay the infomration calculation')

    parser.add_argument('-local_mode',
                        '-lm', dest="local_mode", default=1,
                        help='1 if run it from local computer')

    parser.add_argument('-cov_net',
                        '-cov', dest="cov_net", default=True,
                        help='1 if run it from local computer')

    parser.add_argument('-rl',
                        '-rand_labels', dest="random_labels", default=False,
                        help='')


    args = parser.parse_args()
    args.inds = [map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", args.inds)]
    if num_of_samples != None:
        args.inds = [[num_of_samples]]
    return args


class informationNetwork():
    def save_data(self):
        data = {'information': self.information, 'information_estimation': self.information_estimation,
                     'test_error': self.test_error, 'train_error': self.train_error, 'var_grad_val': self.grads,
                     'loss_test': self.loss_test, 'loss_train': self.loss_train, 'params': self.params
            , 'l1_norms': self.l1_norms, 'weights': self.ws}

        """Save the given data in the given directory"""
        directory = 'jobs/' + self.params['directory'] + '/'
        print directory
        if int(self.args.local_mode)  ==1:
            directory = '/Users/ravidziv/PycharmProjects/IDNNs/jobs/'+self.params['directory'] + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + 'data.pickle', 'wb') as f:
            cPickle.dump(data, f, protocol=2)
        # Also save the code file
        #file_name = os.path.realpath(__file__)
        #srcfile = os.getcwd() + '/' + os.path.basename(file_name)
        #dstdir = directory + os.path.basename(file_name)
        #shutil.copy(srcfile, dstdir)

    def select_network_arch(self,type_net):
        """Selcet the architectures of the networks according to thier type
        Can we choose also costume network"""
        if type_net == '1':
            self.layers_sizes = [[10, 7, 5,4,3]]
            #self.layers_sizes = [[30, 20, 7,5,4]]
        elif type_net == '1-2-3':
            self.layers_sizes = [[10, 9, 7, 7, 3], [10, 9, 7, 5, 3], [10, 9, 7, 3, 3]]
        elif type_net == '11':
            self.layers_sizes = [[10, 7, 7, 4, 3]]
        elif type_net == '2':
            self.layers_sizes = [[10, 7, 5, 4]]
        elif type_net == '3':
            self.layers_sizes = [[10, 7, 5]]
        elif type_net == '4':
            self.layers_sizes = [[10, 7]]
        elif type_net == '5':
            self.layers_sizes = [[10]]
        elif type_net == '6':
            #self.layers_sizes = [[50, 25, 20,10]]
            self.layers_sizes = [[1000, 500, 100]]

        else:
            self.layers_sizes = [map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", type_net)]


    def create_regular_array(self):
        return np.zeros(
            [self.args.num_of_repeats, len(self.layers_sizes), len(self.train_samples), len(self.epochs_indexes)])


    def __init__(self, rand_int = 0, num_of_samples = None, args = None):
        if args ==None:
            args = get_default_parser(num_of_samples)
        self.args = args
        self.name = 'data/' + self.args.data_name
        if int(self.args.local_mode)  ==1:
            self.name = '/Users/ravidziv/PycharmProjects/IDNNs/data/'+self.args.data_name

        # The arch of the networks
        self.select_network_arch(self.args.net_type)
        # The percents of the train data samples
        self.train_samples = np.linspace(1, 100, 199)[[[x * 2 - 2 for x in index] for index in args.inds]]
        # The indexs that we want to calculate the information for them
        self.epochs_indexes = np.unique(
            np.logspace(np.log2(self.args.start_samples), np.log2(self.args.num_ephocs), args.num_of_samples, dtype=int, base=2)) - 1
        max_size = np.max([len(layers_size) for layers_size in self.layers_sizes])
        # load data
        if int(self.args.random_labels) ==1:
            self.args.random_labels = True
        self.data_sets = nn.load_data(self.name, self.args.random_labels)

        #self.name_to_save = args.name_to_store + "_" + "_".join([str(i) + '=' + str(params[i]) for i in params])
        # The weights
        #self.weights = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))] for i in
        #           range(args.num_of_repeats)] if args.save_ws == 1 else 0
        #the variance of the gradints
        #self.var_grads = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))] for i in
        #             range(args.num_of_repeats)] if args.save_grads == 1 else 0
        self.information = np.zeros(
            [args.num_of_repeats, len(self.layers_sizes), len(self.train_samples), len(self.epochs_indexes), max_size + 1,
             2]) if args.calc_information == 1 else 0
        self.information_estimation = np.zeros(
            [args.num_of_repeats, len(self.layers_sizes), len(self.train_samples), len(self.epochs_indexes), max_size + 1,
             2]) if args.calc_information == 1 else 0

        self.ws = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))] for i in
                   range(args.num_of_repeats)]

        self.grads = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))] for i in
                   range(args.num_of_repeats)]

        self.information = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))] for i in
                   range(args.num_of_repeats)]

        self.model = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))]
                            for i in
                            range(args.num_of_repeats)]

        self.names = [[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))]
                      for i in
                      range(args.num_of_repeats)]
        self.loss_train, self.loss_test,  self.test_error, self.train_error, self.l1_norms, self.l2_norms= \
            self.create_regular_array(), self.create_regular_array(), self.create_regular_array(), self.create_regular_array(),self.create_regular_array(), \
            self.create_regular_array()

        params = {'samples_len': len(self.train_samples), 'num_of_disribuation_samples': self.args.num_of_disribuation_samples,
                  'layersSizes': self.layers_sizes, 'numEphocs': self.args.num_ephocs, 'batch': self.args.batch_size,
                  'numRepeats': self.args.num_of_repeats, 'numEpochsInds': len(self.epochs_indexes),
                  'LastEpochsInds': self.epochs_indexes[-1], 'DataName': self.args.data_name, 'learningRate': self.args.learning_rate}
        self.name_to_save = self.args.name + "_" + "_".join([str(i) + '=' + str(params[i]) for i in params])
        params['train_samples'], params['CPUs'], params[
            'directory'],params['epochsInds']  = self.train_samples, NUM_CORES, self.name_to_save,self.epochs_indexes
        self.params =params
        self.rand_int = rand_int


    def run_network(self):
        if self.args.run_in_parallel == 1:
            results = Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
                                                     (self.layers_sizes[j],
                                                      self.args.num_ephocs, self.args.learning_rate, self.args.batch_size,
                                                      self.epochs_indexes, self.args.save_grads, self.data_sets, self.args.model_type,
                                                      self.train_samples,self.args.interval_accuracy_display, self.rand_int)
                                                     for i in range(len(self.train_samples)) for j in
                                                     range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats))
        else:
            results = [nn.train_and_calc_inf_network(i, j,k,
                                                     self.layers_sizes[j],
                                                     self.args.num_ephocs, self.args.learning_rate, self.args.batch_size,
                                                     self.epochs_indexes, self.args.save_grads, self.data_sets, self.args.model_type,
                                                     self.train_samples, self.args.interval_accuracy_display, self.args.calc_information,
                                                     self.args.calc_information_last,
                                                     self.args.num_of_bins, self.args.interval_information_display, self.args.save_ws,
                                                     self.rand_int, self.args.cov_net)
                for i in range(len(self.train_samples))
                           for i in range(len(self.train_samples)) for j in range(len(self.layers_sizes)) for k in
                           range(self.args.num_of_repeats)]
        # Extract all the measures and orgainze it
        for i in range(len(self.train_samples)):
            for j in range(len(self.layers_sizes)):
                for k in range(self.args.num_of_repeats):

                    index = i * len(self.layers_sizes) * self.args.num_of_repeats + j * self.args.num_of_repeats + k
                    information, ws, self.test_error[k, j, i, :], self.train_error[k, j, i, :], self.loss_test[k, j, i, :], self.loss_train[k,j, i,:], \
                    model, name, grads = results[index]
                    self.ws[k][j][i] =ws
                    self.information[k][j][i] = information
                    self.model[k][i][i] = model
                    self.names[k][i][i] = name

                    self.grads[k][i][i] = grads


    def print_information(self):
        for val in self.params:
            if val!='epochsInds':
                print val, self.params[val]

    def inferance(self, data):
        current_model = self.model[0][0][0]
        with tf.Session() as sess:
            current_model.saver.restore(sess, './'+current_model.save_file)
            feed_dict = {current_model.x: data}
            pred = sess.run(current_model.prediction,feed_dict = feed_dict)
        return pred
            #print ('prediction -  {0}'.format(sess.run(current_model.prediction,feed_dict = feed_dict)))

    def calc_information(self):
        #self.information = np.array(Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
        self.information =  np.array([inn.get_infomration(self.ws[k][j][i], self.data_sets.data, self.data_sets.labels,
                                                         self.args.num_of_bins, self.args.interval_information_display, self.epochs_indexes)
                                     for i in range(len(self.train_samples)) for j in
                                     range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])

    def calc_information_last(self):
        wsf = self.ws[0][0][0]
        return np.array([inn.get_infomration([self.ws[k][j][i][-1]], self.data_sets.data, self.data_sets.labels,
                                                         self.args.num_of_bins, self.args.interval_information_display, self.epochs_indexes)
                                     for i in range(len(self.train_samples)) for j in
                                     range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])

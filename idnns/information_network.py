import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import network as nn
import os
import cPickle
import shutil
import re
import argparse
import sys
from idnns.plots import plot_figures as plt_fig
from idnns.information import information_process  as inn
import  tensorflow as tf
NUM_CORES = multiprocessing.cpu_count()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1') or v==True:
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
                        '-i', dest="inds", default='[80]',
                        help='The percent of the training data')

    parser.add_argument('-name',
                        '-na', dest="name", default='net',
                        help='The name to save the results')

    parser.add_argument('-d_name',
                        '-dna', dest="data_name", default='var_u',
                        help='The dataset that we want to run ')

    parser.add_argument('-num_samples',
                        '-ns', dest="num_of_samples", default=1800,
                        type=int,
                        help='The max number of indexes for calculate information')

    parser.add_argument('-num_of_disribuation_samples',
                        '-nds', dest="num_of_disribuation_samples", default=1,
                        type=int,help='S')

    parser.add_argument('-save_ws',
                        '-sws', dest="save_ws", type=str2bool, nargs='?', const=False,default=False,
                        help='if we want to save the output of the layers')

    parser.add_argument('-calc_information',
                        '-cinf', dest="calc_information", type=str2bool, nargs='?', const=True,default=True,
                        help='if we want to calculate the MI in the network for all the epochs')

    parser.add_argument('-calc_information_last',
                        '-cinfl', dest="calc_information_last", type=str2bool, nargs='?', const=False,default=False,
                        help='if we want to calculate the MI in the network only for the last epoch')

    parser.add_argument('-save_grads',
                        '-sgrad', dest="save_grads",type=str2bool, nargs='?', const=False,default=False,
                        help='if we want to save the gradients in the network')

    parser.add_argument('-run_in_parallel',
                        '-par', dest="run_in_parallel", type=str2bool, nargs='?', const=False,default=False,
                        help='If we want to run all the networks in parallel mode')

    parser.add_argument('-num_of_bins',
                        '-nbins', dest="num_of_bins", default=30, type=int,
                        help='The number of bins that we divide the output of the neurons')

    parser.add_argument('-activation_function',
                        '-af', dest="activation_function", default=0, type= int,
                        help='The activation function of the model 0 for thnh 1 for RelU')

    parser.add_argument('-iad',dest="interval_accuracy_display", default=499,type = int,
                        help='The interval for display accuracy')

    parser.add_argument('-interval_information_display',
                        '-iid', dest="interval_information_display", default=30,type = int,
                        help='The interval for display the information calculation')

    parser.add_argument('-cov_net',
                        '-cov', dest="cov_net", type=int, default=0,
                        help='True if we want covnet')

    parser.add_argument('-rl',
                        '-rand_labels', dest="random_labels", type=str2bool, nargs='?', const=False,default=False,
                        help='True if we want to set random labels')
    parser.add_argument('-data_dir',
                        '-dd', dest="data_dir", default='data/',
                        help='The directory for finding the data')


    args = parser.parse_args()
    args.inds = [map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", args.inds)]
    if num_of_samples != None:
        args.inds = [[num_of_samples]]
    return args


class informationNetwork():

    def __init__(self, rand_int = 0, num_of_samples = None, args = None):
        if args ==None:
            args = get_default_parser(num_of_samples)
        self.cov_net = args.cov_net
        self.calc_information = args.calc_information
        self.run_in_parallel = args.run_in_parallel
        self.num_ephocs = args.num_ephocs
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.activation_function = args.activation_function
        self.interval_accuracy_display = args.interval_accuracy_display
        self.save_grads = args.save_grads
        self.num_of_repeats = args.num_of_repeats
        self.calc_information_last = args.calc_information_last
        self.num_of_bins = args.num_of_bins
        self.interval_information_display = args.interval_information_display
        self.save_ws = args.save_ws

        self.name = args.data_dir + args.data_name
        # The arch of the networks
        self.select_network_arch(args.net_type)
        # The percents of the train data samples
        self.train_samples = np.linspace(1, 100, 199)[[[x * 2 - 2 for x in index] for index in args.inds]]
        # The indexs that we want to calculate the information for them in logspace interval
        self.epochs_indexes = np.unique(
            np.logspace(np.log2(args.start_samples), np.log2(args.num_ephocs), args.num_of_samples, dtype=int, base=2)) - 1
        max_size = np.max([len(layers_size) for layers_size in self.layers_sizes])
        #load data
        self.data_sets = nn.load_data(self.name, args.random_labels)
        #create arrays for saving the data
        self.ws, self.grads, self.information,self.models ,self.names, self.networks,self.weights = [[[[[None] for k in range(len(self.train_samples))] for j in range(len(self.layers_sizes))]
                      for i in range(self.num_of_repeats)] for _ in range(7)]

        self.loss_train, self.loss_test,  self.test_error, self.train_error, self.l1_norms, self.l2_norms= \
            [np.zeros((self.num_of_repeats, len(self.layers_sizes), len(self.train_samples), len(self.epochs_indexes))) for _ in range(6)]

        params = {'samples_len': len(self.train_samples), 'num_of_disribuation_samples': args.num_of_disribuation_samples,
                  'layersSizes': self.layers_sizes, 'numEphocs': args.num_ephocs, 'batch': args.batch_size,
                  'numRepeats': args.num_of_repeats, 'numEpochsInds': len(self.epochs_indexes),
                  'LastEpochsInds': self.epochs_indexes[-1], 'DataName': args.data_name, 'learningRate': args.learning_rate}

        self.name_to_save = args.name + "_" + "_".join([str(i) + '=' + str(params[i]) for i in params])

        params['train_samples'], params['CPUs'], params[
            'directory'],params['epochsInds']  = self.train_samples, NUM_CORES, self.name_to_save,self.epochs_indexes
        self.params =params
        self.rand_int = rand_int

        #If we trained already the network
        self.traind_network = False

    def save_data(self,parent_dir='jobs/', file_to_save = 'data.pickle'):
        """Save the data to the file """
        directory =  '{0}/{1}{2}/'.format(os.path.dirname(sys.argv[0]), parent_dir, self.params['directory'])

        data = {'information': self.information,
                     'test_error': self.test_error, 'train_error': self.train_error, 'var_grad_val': self.grads,
                     'loss_test': self.loss_test, 'loss_train': self.loss_train, 'params': self.params
            , 'l1_norms': self.l1_norms, 'weights': self.weights, 'ws': self.ws}

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.dir_saved = directory
        with open(self.dir_saved +file_to_save , 'wb') as f:
            cPickle.dump(data, f, protocol=2)

    def select_network_arch(self,type_net):
        """Selcet the architectures of the networks according to their type
        we can choose also costume network for example type_net=[size_1, size_2, size_3]"""
        if type_net == '1':
            self.layers_sizes = [[10, 7, 5,4,3]]
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
            self.layers_sizes = [[1, 1, 1,1]]
        else:
            #Custom network
            self.layers_sizes = [map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", type_net)]

    def run_network(self):
        """Train and calculated the network's information"""
        if self.run_in_parallel:
            results = Parallel(n_jobs=NUM_CORES)(delayed(nn.train_network)
                                                     (self.layers_sizes[j],
                                                      self.num_ephocs, self.learning_rate, self.batch_size,
                                                      self.epochs_indexes, self.save_grads, self.data_sets, self.activation_function,
                                                      self.train_samples,self.interval_accuracy_display, self.calc_information,
                                                      self.calc_information_last, self.num_of_bins,
                                                      self.interval_information_display, self.save_ws, self.rand_int,self.cov_net)
                                                     for i in range(len(self.train_samples)) for j in
                                                     range(len(self.layers_sizes)) for k in range(self.num_of_repeats))

        else:
            results = [nn.train_and_calc_inf_network(i, j,k,
                                                     self.layers_sizes[j],
                                                     self.num_ephocs, self.learning_rate, self.batch_size,
                                                     self.epochs_indexes, self.save_grads, self.data_sets, self.activation_function,
                                                     self.train_samples, self.interval_accuracy_display, self.calc_information,
                                                     self.calc_information_last, self.num_of_bins, self.interval_information_display,
                                                     self.save_ws, self.rand_int, self.cov_net)
                for i in range(len(self.train_samples))
                           for i in range(len(self.train_samples)) for j in range(len(self.layers_sizes)) for k in
                           range(self.num_of_repeats)]

        # Extract all the measures and orgainze it
        for i in range(len(self.train_samples)):
            for j in range(len(self.layers_sizes)):
                for k in range(self.num_of_repeats):
                    index = i * len(self.layers_sizes) * self.num_of_repeats + j * self.num_of_repeats + k
                    current_network = results[index]
                    self.networks[k][j][i] = current_network
                    self.ws[k][j][i] =current_network['ws']
                    self.weights[k][j][i] = current_network['weights']
                    self.information[k][j][i] = current_network['information']
                    self.grads[k][i][i] = current_network['gradients']
                    self.test_error[k, j, i, :] = current_network['test_prediction']
                    self.train_error[k, j, i, :] = current_network['train_prediction']
                    self.loss_test[k, j ,i, :] =  current_network['loss_test']
                    self.loss_train[k, j, i, :] =  current_network['loss_train']

        self.traind_network = True

    def print_information(self):
        """Print the networks params"""
        for val in self.params:
            if val!='epochsInds':
                print val, self.params[val]


    def calc_information(self):
        """Calculate the infomration of the network for all the epochs - only valid if we save the activation values and trained the network"""
        if self.traind_network and self.save_ws:
            self.information =  np.array([inn.get_information(self.ws[k][j][i], self.data_sets.data, self.data_sets.labels,
                                                              self.args.num_of_bins, self.args.interval_information_display, self.epochs_indexes)
                                         for i in range(len(self.train_samples)) for j in
                                         range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])
        else:
            print ('Cant calculate the infomration of the networks!!!')

    def calc_information_last(self):
        """Calculate the information of the last epoch"""
        if self.traind_network and self.save_ws:
            return np.array([inn.get_information([self.ws[k][j][i][-1]], self.data_sets.data, self.data_sets.labels,
                                                 self.args.num_of_bins, self.args.interval_information_display, self.epochs_indexes)
                                     for i in range(len(self.train_samples)) for j in
                                     range(len(self.layers_sizes)) for k in range(self.args.num_of_repeats)])

    def plot_network(self):
        str_names = [[self.dir_saved]]
        mode =2
        save_name = 'figure'
        plt_fig.plot_figures(str_names, mode, save_name)

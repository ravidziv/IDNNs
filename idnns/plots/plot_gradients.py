'Calculate and plot the gradients (the mean and std of the mini-batch gradients) of the trained network'
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import idnns.plots.utils as plt_ut
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from numpy import linalg as LA
import os
import sys
import statsmodels
colors = ['red', 'c', 'blue', 'green', 'orange', 'purple']


def plot_gradients(name_s=None, data_array=None, figures_dir=''):
    """Plot the gradients and the means of the networks over the batches"""
    if data_array == None:
        data_array= plt_ut.get_data(name_s[0][0])
    #plot_loss_figures(data_array, xlim = [0, 7000] )
    #The gradients - the diemnstions are #epochs X #Batchs # Layers
    conv_net = False
    if conv_net:
        gradients =data_array['var_grad_val'][0][0][0]
        num_of_epochs = len(gradients)
        num_of_batchs = len(gradients[0])
        num_of_layers = len(gradients[0][0]) / 2
    else:
        gradients = np.squeeze(data_array['var_grad_val'])[:, :, :]
        num_of_epochs,num_of_batchs,  num_of_layers = gradients.shape
        num_of_layers = int(num_of_layers / 2)
    #The indxes where we sampled the network
    print (np.squeeze(data_array['var_grad_val'])[0,0].shape)
    epochsInds = (data_array['params']['epochsInds']).astype(np.int)
    #The norms of the layers
    #l2_norm = calc_weights_norms(data_array['ws_all'])
    f_log, axes_log, f_norms, axes_norms, f_snr,  axes_snr,axes_gaus, f_gaus = create_figs()
    p_1, p_0,  sum_y ,p_3, p_4= [], [], [], [], []
    # Go over the layers
    cov_traces_all,means_all = [],[]
    all_gradients = np.empty(num_of_layers, dtype=np.object)
    #print np.squeeze(data_array['var_grad_val']).shape
    for layer in range(0,num_of_layers):
        # The traces of the covarince and the means of the gradients for the current layer
        # Go over all the epochs
        cov_traces, means = [], []
        gradinets_layer = []
        for epoch_index in range(num_of_epochs):
            # the gradients are dimensions of #batchs X # output weights - when output weights is the number of wieghts that go out from the layer
            gradients_current_epoch_and_layer = flatted_graidnet(gradients, epoch_index, 2 * layer)
            gradinets_layer.append(gradients_current_epoch_and_layer)
            num_of_output_weights =  gradients_current_epoch_and_layer.shape[1]
            # the average vector over the batchs - this is vector in the size of #output weights
            # We averged over the batchs - It's mean vector of the batchs!
            average_vec = np.mean(gradients_current_epoch_and_layer, axis=0)
            # The sqrt of the sum over all the weights of the squares of the gradinets -  Sqrt of AA^T - This is a number
            gradients_mean = LA.norm(average_vec)
            # The covarince matrix is in the size of #output weights X #output weights
            sum_covs_mat = np.zeros((average_vec.shape[0], average_vec.shape[0]))
            # Go over all the vectors of batchs (each vector is the size of # output weights, reduce the mean (over the batchs)
            # and calculate the covariance matrix
            for batch_index in range(num_of_batchs):
                # This is in the size of the #output weights
                current_vec = gradients_current_epoch_and_layer[batch_index, :] - average_vec
                # The outer product of the current gradinet of the weights (in this specipic batch) with the transpose of it -
                # give a matrix in the size of # output weights X # output weights
                current_cov_mat = np.einsum('i,j', current_vec, current_vec)
                #current_cov_mat = np.dot(current_vec[:,None], current_vec[None,:])
                # Sum the covarince matrixes over the batchs
                sum_covs_mat+=current_cov_mat
            #Take the mean of the cov matrix over the batchs  - The size is #output weights X # output weights
            mean_cov_mat = sum_covs_mat / num_of_batchs
            #The trace of the mean of the cov matrix - a number
            trac_cov = np.sqrt(np.trace(mean_cov_mat))
            means.append(gradients_mean)
            cov_traces.append(trac_cov)
            """

                #cov_traces.append(np.mean(grad_norms))
                #means.append(norm_mean)
                c_var,c_mean,total_w = [], [],[]

                for neuron in range(len(grad[epoch_number][0][layer])/10):
                    gradients_list = np.array([grad[epoch_number][i][layer][neuron] for i in range(len(grad[epoch_number]))])
                    total_w.extend(gradients_list.T)
                    grad_norms1 = np.std(gradients_list, axis=0)
                    mean_la = np.abs(np.mean(np.array(gradients_list), axis=0))
                    #mean_la = LA.norm(gradients_list, axis=0)
                    c_var.append(np.mean(grad_norms1))
                    c_mean.append(np.mean(mean_la))
                #total_w is in size [num_of_total_weights, num of epochs]
                total_w = np.array(total_w)
                #c_var.append(np.sqrt(np.trace(np.cov(np.array(total_w).T)))/np.cov(np.array(total_w).T).shape[0])
                #print np.mean(c_mean).shape
                means.append(np.mean(c_mean))
                cov_traces.append(np.mean(c_var))
            """

        gradinets_layer = np.array(gradinets_layer)
        all_gradients[layer]= gradinets_layer
        cov_traces_all.append(np.array(cov_traces))
        means_all.append(np.array(means))
        #The cov_traces and the means are vectors with the dimension of # epochs
        #y_var = np.array(cov_traces)
        #y_mean = np.array(means)
        y_var = np.sum(cov_traces_all, axis=0)
        y_mean = np.sum(means_all, axis=0)
        snr =  y_mean**2 / y_var
        #Plot the gradients and the means
        c_p1, = axes_log.plot(epochsInds[:], np.sqrt(y_var),markersize = 4, linewidth = 4,color = colors[layer], linestyle=':', markeredgewidth=0.2, dashes = [4,4])
        c_p0,= axes_log.plot(epochsInds[:], y_mean,  linewidth = 2,color = colors[layer])
        c_p3,= axes_snr.plot(epochsInds[:],snr,  linewidth = 2,color = colors[layer])
        c_p4,= axes_gaus.plot(epochsInds[:],np.log(1+snr),  linewidth = 2,color = colors[layer])
        #For the legend
        p_0.append(c_p0), p_1.append(c_p1),sum_y.append(y_mean) , p_3.append(c_p3), p_4.append(c_p4)
    plt_ut.adjust_axes(axes_log, axes_norms, p_0, p_1, f_log, f_norms, axes_snr, f_snr, p_3, axes_gaus, f_gaus, p_4,
                       directory_name=figures_dir)
    plt.show()


def calc_mean_var_loss(epochsInds,loss_train):
    #Loss train is in dimension # epochs X #batchs
    num_of_epochs = loss_train.shape[0]
    #Average over the batchs
    loss_train_mean = np.mean(loss_train,1)
    #The diff divided by the sampled indexes
    d_mean_loss_to_dt = np.sqrt(np.abs(np.diff(loss_train_mean) / np.diff(epochsInds[:])))
    var_loss = []
    #Go over the epochs
    for epoch_index in range(num_of_epochs):
        #The loss for the specpic epoch
        current_loss = loss_train[epoch_index, :]
        #The derivative between the batchs
        current_loss_dt = np.diff(current_loss)
        #The mean of his derivative
        average_loss = np.mean(current_loss_dt)
        current_loss_minus_mean = current_loss_dt- average_loss
        #The covarince between the batchs
        cov_mat = np.dot(current_loss_minus_mean[:, None], current_loss_minus_mean[None, :])
        # The trace of the cov matrix
        trac_cov = np.trace(cov_mat)
        var_loss.append(trac_cov)
    return np.array(var_loss), d_mean_loss_to_dt

def plot_loss_figures(data_array, fig_size=(14, 10), xlim = None, y_lim = None):
    epochsInds = (data_array['params']['epochsInds']).astype(np.int)
    dif_var_loss, diff_mean_loss = calc_mean_var_loss(epochsInds, np.squeeze(data_array['loss_train']))
    f_log1, (axes_log1) = plt.subplots(1, 1, figsize=fig_size)
    axes_log1.set_title('The mean and the varince( between the batchs) of the derivative of the train error')
    axes_log1.plot(epochsInds[1:], np.array(diff_mean_loss), color='green', label = 'Mean of the derivative of the error')
    axes_log1.plot(epochsInds[:], (dif_var_loss), color='blue', label='Variance of the derivative of the error' )
    axes_log1.set_xscale('log')
    axes_log1.set_yscale('log')
    axes_log1.set_xlabel('#Epochs')
    axes_log1.legend()

    f_log1, (axes_log1) = plt.subplots(1, 1, figsize=fig_size)
    title = r'The SNR of the error derivatives'

    p_5, =axes_log1.plot(epochsInds[1:], np.array(diff_mean_loss)/ np.sqrt(dif_var_loss[1:]), linewidth = 3, color='green',
                  )
    plt_ut.update_axes(axes_log1, f_log1, '#Epochs', 'SNR',[0, 7000], [0.001, 1], title, 'log', 'log',
                       [1, 10, 100, 1000, 7000], [0.001, 0.01, 0.1, 1])
    #axes_log1.plot(epochsInds[:], (dif_var_loss), color='blue', label='Variance of the derivative of the error')
    axes_log1.legend([r'$\frac{|d Error|}{STD\left(Error)\right)}$'], loc= 'best',fontsize = 21)

def create_figs(fig_size = (14, 10)):
    f_norms, (axes_norms) = plt.subplots(1, 1, figsize=fig_size)
    f_log, (axes_log) = plt.subplots(1, 1, figsize=fig_size)
    f_snr, (axes_snr) = plt.subplots(1, 1, figsize=fig_size)
    f_gaus, (axes_gaus) = plt.subplots(1, 1, figsize=fig_size)
    f_log.subplots_adjust(left=0.097, bottom=0.11, right=.95, top=0.95, wspace=0.03, hspace=0.03)
    return f_log, axes_log, f_norms, axes_norms, f_snr, axes_snr,axes_gaus, f_gaus


def flatted_graidnet(gradients, epoch_number, layer):
    gradients_list = []
    # For each neuron in the current layer go over all the weights
    for i in range(len(gradients[epoch_number])):
        current_list_inner = []
        for neuron in range(len(gradients[epoch_number][0][layer])):
            c_n = gradients[epoch_number][i][layer][neuron]
            current_list_inner.extend(c_n)
        gradients_list.append(current_list_inner)
    gradients_list = np.array(gradients_list)
    gradients_list =np.reshape(gradients_list, (gradients_list.shape[0], -1))

    return gradients_list


def calc_weights_norms(ws, num_of_layer = 6):
    layer_l2_norm = []
    for i in range(num_of_layer):
        flatted_list = [1]
        """
        if type(ws_in[epoch_number][layer_index]) is list:
            flatted_list = [item for sublist in ws_in[epoch_number][layer_index] for item in sublist]
        else:
            flatted_list = ws_in[epoch_number][layer_index]
        """
        layer_l2_norm.append(LA.norm(np.array(flatted_list)))
    # plot the norms
    #axes_norms.plot(epochsInds[:], np.array(layer_l2_norm), linewidth=2, color=colors[layer_index])
    return layer_l2_norm

def extract_array(data, name):
    results = [[data[j,k][name] for k in range(data.shape[1])] for j in range(data.shape[0])]
    return results


def load_from_memory(data_array):
    plot_gradients(data_array=data_array)


if __name__ == '__main__':
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    str_names = [[('/').join(file_path.split('/')[:-1]) + '/']]
    plot_gradients(str_names, figures_dir=directory)

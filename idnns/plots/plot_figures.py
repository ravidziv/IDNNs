"""Plot the networks in the information plane"""
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import _pickle as cPickle
# import cPickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import scipy.stats as sis
import os
import matplotlib.animation as animation
import math
import os.path
import idnns.plots.utils as utils
import tkinter as tk
from numpy import linalg as LA

from tkinter import filedialog
LAYERS_COLORS  = ['red', 'blue', 'green', 'yellow', 'pink', 'orange']

def plot_all_epochs(gen_data, I_XT_array, I_TY_array, axes, epochsInds, f, index_i, index_j, size_ind,
                    font_size, y_ticks, x_ticks, colorbar_axis, title_str, axis_font, bar_font, save_name, plot_error = True,index_to_emphasis=1000):
    """Plot the infomration plane with the epochs in diffrnet colors """
    #If we want to plot the train and test error
    if plot_error:
        fig_strs = ['train_error','test_error','loss_train','loss_test' ]
        fig_data = [np.squeeze(gen_data[fig_str]) for fig_str in fig_strs]
        f1 = plt.figure(figsize=(12, 8))
        ax1 = f1.add_subplot(111)
        mean_sample = False if len(fig_data[0].shape)==1 else True
        if mean_sample:
            fig_data = [ np.mean(fig_data_s, axis=0) for fig_data_s in fig_data]
        for i in range(len(fig_data)):
            ax1.plot(epochsInds, fig_data[i],':', linewidth = 3 , label = fig_strs[i])
        ax1.legend(loc='best')
    f = plt.figure(figsize=(12, 8))
    axes = f.add_subplot(111)
    axes = np.array([[axes]])

    I_XT_array = np.squeeze(I_XT_array)
    I_TY_array = np.squeeze(I_TY_array)
    if len(I_TY_array[0].shape) >1:
        I_XT_array = np.mean(I_XT_array, axis=0)
        I_TY_array = np.mean(I_TY_array, axis=0)
    max_index = size_ind if size_ind != -1 else I_XT_array.shape[0]

    cmap = plt.get_cmap('gnuplot')
    #For each epoch we have diffrenet color
    colors = [cmap(i) for i in np.linspace(0, 1, epochsInds[max_index-1]+1)]
    #Change this if we have more then one network arch
    nums_arc= -1
    #Go over all the epochs and plot then with the right color
    for index_in_range in range(0, max_index):
        XT = I_XT_array[index_in_range, :]
        TY = I_TY_array[index_in_range, :]
        #If this is the index that we want to emphsis
        if epochsInds[index_in_range] ==index_to_emphasis:
            axes[index_i, index_j].plot(XT, TY, marker='o', linestyle=None, markersize=19, markeredgewidth=0.04,
                                        linewidth=2.1,
                                        color='g',zorder=10)
        else:
                axes[index_i, index_j].plot(XT[:], TY[:], marker='o', linestyle='-', markersize=12, markeredgewidth=0.01, linewidth=0.2,
                                color=colors[int(epochsInds[index_in_range])])
    utils.adjustAxes(axes[index_i, index_j], axis_font=axis_font, title_str=title_str, x_ticks=x_ticks,
                     y_ticks=y_ticks, x_lim=[0, 25.1], y_lim=None,
                     set_xlabel=index_i == axes.shape[0] - 1, set_ylabel=index_j == 0, x_label='$I(X;T)$',
                     y_label='$I(T;Y)$', set_xlim=False,
                     set_ylim=False, set_ticks=True, label_size=font_size)
    #Save the figure and add color bar
    if index_i ==axes.shape[0]-1 and index_j ==axes.shape[1]-1:
        utils.create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title='Epochs')
        f.savefig(save_name+'.jpg', dpi=500, format='jpg')


def plot_by_training_samples(I_XT_array, I_TY_array, axes, epochsInds, f, index_i, index_j, size_ind, font_size, y_ticks, x_ticks, colorbar_axis, title_str, axis_font, bar_font, save_name, samples_labels):
    """Print the final epoch of all the diffrenet training samples size """
    max_index = size_ind if size_ind!=-1 else I_XT_array.shape[2]-1
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, max_index+1)]
    #Print the final epoch
    nums_epoch= -1
    #Go over all the samples size and plot them with the right color
    for index_in_range in range(0, max_index):
        XT, TY = [], []
        for layer_index in range(0, I_XT_array.shape[4]):
                XT.append(np.mean(I_XT_array[:, -1, index_in_range, nums_epoch, layer_index], axis=0))
                TY.append(np.mean(I_TY_array[:, -1, index_in_range,nums_epoch, layer_index], axis=0))
        axes[index_i, index_j].plot(XT, TY, marker='o', linestyle='-', markersize=12, markeredgewidth=0.2, linewidth=0.5,
                         color=colors[index_in_range])
    utils.adjustAxes(axes[index_i, index_j], axis_font=axis_font, title_str=title_str, x_ticks=x_ticks, y_ticks=y_ticks,
                     x_lim=None, y_lim=None,
                     set_xlabel=index_i == axes.shape[0] - 1, set_ylabel=index_j == 0, x_label='$I(X;T)$',
                     y_label='$I(T;Y)$', set_xlim=True,
                     set_ylim=True, set_ticks=True, label_size=font_size)
    #Create color bar and save it
    if index_i == axes.shape[0] - 1 and index_j == axes.shape[1] - 1:
        utils.create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title='Training Data')
        f.savefig(save_name + '.jpg', dpi=150, format='jpg')

def calc_velocity(data, epochs):
    """Calculate the velocity (both in X and Y) for each layer"""
    vXs, vYs = [], []
    for layer_index in range(data.shape[5]):
        curernt_vXs = []
        current_VYs = []
        for epoch_index in range(len(epochs)-1):
            vx = np.mean(data[0,:,-1, -1, epoch_index+1,layer_index], axis=0) - np.mean(data[0,:,-1, -1, epoch_index,layer_index], axis=0)
            vx/= (epochs[epoch_index+1] - epochs[epoch_index])
            vy = np.mean(data[1, :, -1, -1, epoch_index + 1, layer_index], axis=0) - np.mean(data[1, :, -1, -1, epoch_index, layer_index],                                                                      axis=0)
            vy /= (epochs[epoch_index + 1] - epochs[epoch_index])
            current_VYs.append(vy)
            curernt_vXs.append(vx)
        vXs.append(curernt_vXs)
        vYs.append(current_VYs)
    return vXs,vYs

def update_line_specipic_points(nums, data, axes, to_do, font_size, axis_font):
    """Update the lines in the axes for snapshot of the whole process"""
    colors =LAYERS_COLORS
    x_ticks = [0, 2, 4, 6, 8, 10]
    #Go over all the snapshot
    for i in range(len(nums)):
        num = nums[i]
        #Plot the right layer
        for layer_num in range(data.shape[3]):
            axes[i].scatter(data[0, :, num, layer_num], data[1, :, num, layer_num], color = colors[layer_num], s = 105,edgecolors = 'black',alpha = 0.85)
        utils.adjustAxes(axes[i], axis_font=axis_font, title_str='', x_ticks=x_ticks, y_ticks=[], x_lim=None,
                         y_lim=None,
                         set_xlabel=to_do[i][0], set_ylabel=to_do[i][1], x_label='$I(X;T)$', y_label='$I(T;Y)$',
                         set_xlim=True, set_ylim=True,
                         set_ticks=True, label_size=font_size)

def update_line_each_neuron(num, print_loss, Ix, axes, Iy, train_data, accuracy_test, epochs_bins, loss_train_data, loss_test_data, colors, epochsInds,
                            font_size = 18, axis_font = 16, x_lim = [0,12.2], y_lim=[0, 1.08],x_ticks = [], y_ticks = []):
    """Update the figure of the infomration plane for the movie"""
    #Print the line between the points
    axes[0].clear()
    if len(axes)>1:
        axes[1].clear()
    #Print the points
    for layer_num in range(Ix.shape[2]):
        for net_ind in range(Ix.shape[0]):
            axes[0].scatter(Ix[net_ind,num, layer_num], Iy[net_ind,num, layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)
    title_str = 'Information Plane - Epoch number - ' + str(epochsInds[num])
    utils.adjustAxes(axes[0], axis_font, title_str, x_ticks, y_ticks, x_lim, y_lim, set_xlabel=True, set_ylabel=True,
                     x_label='$I(X;T)$', y_label='$I(T;Y)$')
    #Print the loss function and the error
    if len(axes)>1:
        axes[1].plot(epochsInds[:num], 1 - np.mean(accuracy_test[:, :num], axis=0), color='g')
        if print_loss:
            axes[1].plot(epochsInds[:num], np.mean(loss_test_data[:, :num], axis=0), color='y')
        nereast_val = np.searchsorted(epochs_bins, epochsInds[num], side='right')
        axes[1].set_xlim([0,epochs_bins[nereast_val]])
        axes[1].legend(('Accuracy', 'Loss Function'), loc='best')

def update_line(num, print_loss, data, axes, epochsInds, test_error, test_data, epochs_bins, loss_train_data, loss_test_data, colors,
                font_size = 18, axis_font=16, x_lim = [0,12.2], y_lim=[0, 1.08], x_ticks = [], y_ticks = []):
    """Update the figure of the infomration plane for the movie"""
    #Print the line between the points
    cmap = ListedColormap(LAYERS_COLORS)
    segs = []
    for i in range(0, data.shape[1]):
        x = data[0, i, num, :]
        y = data[1, i, num, :]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs.append(np.concatenate([points[:-1], points[1:]], axis=1))
    segs = np.array(segs).reshape(-1, 2, 2)
    axes[0].clear()
    if len(axes)>1:
        axes[1].clear()
    lc = LineCollection(segs, cmap=cmap, linestyles='solid',linewidths = 0.3, alpha = 0.6)
    lc.set_array(np.arange(0,5))
    #Print the points
    for layer_num in range(data.shape[3]):
        axes[0].scatter(data[0, :, num, layer_num], data[1, :, num, layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)
    axes[1].plot(epochsInds[:num], 1 - np.mean(test_error[:, :num], axis=0), color ='r')

    title_str = 'Information Plane - Epoch number - ' + str(epochsInds[num])
    utils.adjustAxes(axes[0], axis_font, title_str, x_ticks, y_ticks, x_lim, y_lim, set_xlabel=True, set_ylabel=True,
                     x_label='$I(X;T)$', y_label='$I(T;Y)$')
    title_str = 'Precision as function of the epochs'
    utils.adjustAxes(axes[1], axis_font, title_str, x_ticks, y_ticks, x_lim, y_lim, set_xlabel=True, set_ylabel=True,
                     x_label='# Epochs', y_label='Precision')


def plot_animation(name_s, save_name):
    """Plot the movie for all the networks in the information plane"""
    # If we want to print the loss function also
    print_loss  = False
    #The bins that we extened the x axis of the accuracy each time
    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]

    data_array = utils.get_data(name_s[0][0])
    data = data_array['infomration']
    epochsInds = data_array['epochsInds']
    loss_train_data = data_array['loss_train']
    loss_test_data = data_array['loss_test_data']
    f, (axes) = plt.subplots(2, 1)
    f.subplots_adjust(left=0.14, bottom=0.1, right=.928, top=0.94, wspace=0.13, hspace=0.55)
    colors = LAYERS_COLORS
    #new/old version
    if False:
        Ix = np.squeeze(data[0,:,-1,-1, :, :])
        Iy = np.squeeze(data[1,:,-1,-1, :, :])
    else:
        Ix = np.squeeze(data[0, :, -1, -1, :, :])[np.newaxis,:,:]
        Iy = np.squeeze(data[1, :, -1, -1, :, :])[np.newaxis,:,:]
    #Interploation of the samplings (because we don't cauclaute the infomration in each epoch)
    interp_data_x = interp1d(epochsInds,  Ix, axis=1)
    interp_data_y = interp1d(epochsInds,  Iy, axis=1)
    new_x = np.arange(0,epochsInds[-1])
    new_data  = np.array([interp_data_x(new_x), interp_data_y(new_x)])
    """"
    train_data = interp1d(epochsInds,  np.squeeze(train_data), axis=1)(new_x)
    test_data = interp1d(epochsInds,  np.squeeze(test_data), axis=1)(new_x)
    """
    if print_loss:
        loss_train_data =  interp1d(epochsInds,  np.squeeze(loss_train_data), axis=1)(new_x)
        loss_test_data=interp1d(epochsInds,  np.squeeze(loss_test_data), axis=1)(new_x)
    line_ani = animation.FuncAnimation(f, update_line, len(new_x), repeat=False,
                                       interval=1, blit=False, fargs=(print_loss, new_data, axes,new_x,train_data,test_data,epochs_bins, loss_train_data,loss_test_data, colors))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100)
    #Save the movie
    line_ani.save(save_name+'_movie2.mp4',writer=writer,dpi=250)
    plt.show()


def plot_animation_each_neuron(name_s, save_name, print_loss=False):
    """Plot the movie for all the networks in the information plane"""
    # If we want to print the loss function also
    #The bins that we extened the x axis of the accuracy each time
    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]
    data_array = utils.get_data(name_s[0][0])
    data = np.squeeze(data_array['information'])

    f, (axes) = plt.subplots(1, 1)
    axes = [axes]
    f.subplots_adjust(left=0.14, bottom=0.1, right=.928, top=0.94, wspace=0.13, hspace=0.55)
    colors = LAYERS_COLORS
    #new/old version
    Ix = np.squeeze(data[0,:, :, :])
    Iy = np.squeeze(data[1,:, :, :])
    #Interploation of the samplings (because we don't cauclaute the infomration in each epoch)
    #interp_data_x = interp1d(epochsInds,  Ix, axis=1)
    #interp_data_y = interp1d(epochsInds,  Iy, axis=1)
    #new_x = np.arange(0,epochsInds[-1])
    #new_data  = np.array([interp_data_x(new_x), interp_data_y(new_x)])
    """"
    train_data = interp1d(epochsInds,  np.squeeze(train_data), axis=1)(new_x)
    test_data = interp1d(epochsInds,  np.squeeze(test_data), axis=1)(new_x)

    if print_loss:
        loss_train_data =  interp1d(epochsInds,  np.squeeze(loss_train_data), axis=1)(new_x)
        loss_test_data=interp1d(epochsInds,  np.squeeze(loss_test_data), axis=1)(new_x)
    """
    line_ani = animation.FuncAnimation(f, update_line_each_neuron, Ix.shape[1], repeat=False,
                                       interval=1, blit=False, fargs=(print_loss, Ix, axes,Iy,train_data,test_data,epochs_bins, loss_train_data,loss_test_data, colors,epochsInds))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100)
    #Save the movie
    line_ani.save(save_name+'_movie.mp4',writer=writer,dpi=250)
    plt.show()


def plot_snapshots(name_s, save_name, i, time_stemps=[13, 180, 963],font_size = 36,axis_font = 28,fig_size = (14, 6)):
    """Plot snapshots of the given network"""
    f, (axes) = plt.subplots(1, len(time_stemps), sharey=True, figsize=fig_size)
    f.subplots_adjust(left=0.095, bottom=0.18, right=.99, top=0.97, wspace=0.03, hspace=0.03)
    #Adding axes labels
    to_do = [[True, True], [True, False], [True, False]]
    data_array = utils.get_data(name_s)
    data = np.squeeze(data_array['information'])
    update_line_specipic_points(time_stemps, data, axes, to_do, font_size, axis_font)
    f.savefig(save_name + '.jpg', dpi=200, format='jpg')


def load_figures(mode, str_names=None):
    """Creaet new figure based on the mode of it
    This function is really messy and need to rewrite """
    if mode == 0:
        font_size = 34
        axis_font = 28
        bar_font = 28
        fig_size = (14, 6.5)
        title_strs = [['', '']]
        f, (axes) = plt.subplots(1, 2, sharey=True, figsize=fig_size)
        sizes = [[-1, -1]]
        colorbar_axis = [0.935, 0.14, 0.027, 0.75]
        axes = np.vstack(axes).T
        f.subplots_adjust(left=0.09, bottom=0.15, right=.928, top=0.94, wspace=0.03, hspace=0.04)
        yticks = [0, 1, 2, 3]
        xticks = [3, 6, 9, 12, 15]
    # for 3 rows with 2 colmus
    if mode == 1:
        font_size = 34
        axis_font = 26
        bar_font = 28
        fig_size = (14, 19)
        xticks = [1, 3, 5, 7, 9, 11]
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        title_strs = [['One hidden layer', 'Two hidden layers'], ['Three hidden layers', 'Four hidden layers'],
                      ['Five hidden layers', 'Six hidden layers']]

        title_strs = [['5 bins', '12 bins'], ['18 bins', '25 bins'],
                      ['35 bins', '50 bins']]
        f, (axes) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=fig_size)
        f.subplots_adjust(left=0.09, bottom=0.08, right=.92, top=0.94, wspace=0.03, hspace=0.15)
        colorbar_axis = [0.93, 0.1, 0.035, 0.76]
        sizes = [[1010, 1010], [1017, 1020], [1700, 920]]

    # for 2 rows with 3 colmus
    if mode == 11:
        font_size = 34
        axis_font = 26
        bar_font = 28
        fig_size = (14, 9)
        xticks = [1, 3, 5, 7, 9, 11]
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        title_strs = [['One hidden layer', 'Two hidden layers','Three hidden layers'], ['Four hidden layers',
                      'Five hidden layers', 'Six hidden layers']]

        title_strs = [['5 Bins', '10 Bins', '15 Bins'], ['20 Bins',
                                                                                         '25 Bins',
                                                                                         '35 Bins']]
        f, (axes) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=fig_size)
        f.subplots_adjust(left=0.09, bottom=0.1, right=.92, top=0.94, wspace=0.03, hspace=0.15)
        colorbar_axis = [0.93, 0.1, 0.035, 0.76]
        sizes = [[1010, 1010, 1017], [1020, 1700, 920]]
    # one figure
    if mode == 2 or mode ==6:
        axis_font = 28
        bar_font = 28
        fig_size = (14, 10)
        font_size = 34
        f, (axes) = plt.subplots(1, len(str_names), sharey=True, figsize=fig_size)
        if len(str_names) == 1:
            axes = np.vstack(np.array([axes]))
        f.subplots_adjust(left=0.097, bottom=0.12, right=.87, top=0.99, wspace=0.03, hspace=0.03)
        colorbar_axis = [0.905, 0.12, 0.03, 0.82]
        xticks = [1, 3, 5, 7, 9, 11]
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

        #yticks = [0, 1, 2, 3, 3.5]
        #xticks = [2, 5, 8, 11, 14, 17]
        sizes = [[-1]]
        title_strs = [['', '']]
    # one figure with error bar
    if mode == 3:
        fig_size = (14, 10)
        font_size = 36
        axis_font = 28
        bar_font = 25
        title_strs = [['']]
        f, (axes) = plt.subplots(1, len(str_names), sharey=True, figsize=fig_size)
        if len(str_names) == 1:
            axes = np.vstack(np.array([axes]))
        f.subplots_adjust(left=0.097, bottom=0.12, right=.90, top=0.99, wspace=0.03, hspace=0.03)
        sizes = [[-1]]
        colorbar_axis = [0.933, 0.125, 0.03, 0.83]
        xticks = [0, 2, 4, 6, 8, 10, 12]
        yticks = [0.3, 0.4, 0.6, 0.8, 1]
        # two figures second
    if mode == 4:
        font_size = 27
        axis_font = 18
        bar_font = 23
        fig_size = (14, 6.5)
        title_strs = [['', '']]
        f, (axes) = plt.subplots(1, 2, figsize=fig_size)
        sizes = [[-1, -1]]
        colorbar_axis = [0.948, 0.08, 0.025, 0.81]
        axes = np.vstack(axes).T
        f.subplots_adjust(left=0.07, bottom=0.15, right=.933, top=0.94, wspace=0.12, hspace=0.04)
        #yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        #xticks = [1, 3, 5, 7, 9, 11]

        yticks = [0,  1,  2, 3, 3]
        xticks = [2, 5, 8, 11,14, 17]
    return font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes


def plot_figures(str_names, mode, save_name):
    """Plot the data in the given names with the given mode"""
    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes] = load_figures(mode, str_names)
    #Go over all the files
    for i in range(len(str_names)):
        for j in range(len(str_names[i])):
            name_s = str_names[i][j]
            #Load data for the given file
            data_array = utils.get_data(name_s)
            data  = np.squeeze(np.array(data_array['information']))
            I_XT_array = np.array(extract_array(data, 'local_IXT'))
            I_TY_array = np.array(extract_array(data, 'local_ITY'))
            #I_XT_array = np.array(extract_array(data, 'IXT_vartional'))
            #I_TY_array = np.array(extract_array(data, 'ITY_vartional'))
            epochsInds = data_array['params']['epochsInds']
            #I_XT_array = np.squeeze(np.array(data))[:, :, 0]
            #I_TY_array = np.squeeze(np.array(data))[:, :, 1]
            #Plot it
            if mode ==3:
                plot_by_training_samples(I_XT_array, I_TY_array, axes, epochsInds, f, i, j, sizes[i][j], font_size, yticks, xticks, colorbar_axis, title_strs[i][j], axis_font, bar_font, save_name)
            elif mode ==6:
                plot_norms(axes, epochsInds,data_array['norms1'],data_array['norms2'])
            else:
                plot_all_epochs(data_array, I_XT_array, I_TY_array, axes, epochsInds, f, i, j, sizes[i][j], font_size, yticks, xticks,
                                colorbar_axis, title_strs[i][j], axis_font, bar_font, save_name)
    plt.show()


def plot_norms(axes, epochsInds, norms1, norms2):
    """Plot the norm l1 and l2 of the given name"""
    axes.plot(epochsInds, np.mean(norms1[:,0,0,:], axis=0), color='g')
    axes.plot(epochsInds, np.mean(norms2[:,0,0,:], axis=0), color='b')
    axes.legend(('L1 norm', 'L2 norm'))
    axes.set_xlabel('Epochs')

def plot_pearson(name):
    """Plot the pearsin coeff of  the neurons for each layer"""
    data_array = utils.get_data(name)
    ws = data_array['weights']
    f = plt.figure(figsize=(12, 8))
    axes = f.add_subplot(111)
    #The number of neurons in each layer -
    #TODO need to change it to be auto
    sizes =[10,7, 5, 4,3,2 ]
    #The mean of pearson coeffs of all the layers
    pearson_mean =[]
    #Go over all the layers
    for layer in range(len(sizes)):
        inner_pearson_mean =[]
        #Go over all the weights in the layer
        for k in range(len(ws)):
            ws_current = np.squeeze(ws[k][0][0][-1])
            #Go over the neurons
            for neuron in range(len(ws_current[layer])):
                person_t = []
                #Go over the rest of the neurons
                for neuron_second in range(neuron+1, len(ws_current[layer])):
                    pearson_c, p_val =sis.pearsonr(ws_current[layer][neuron], ws_current[layer][neuron_second])
                    person_t.append(pearson_c)
            inner_pearson_mean.append(np.mean(person_t))
        pearson_mean.append(np.mean(inner_pearson_mean))
    #Plot the coeff
    axes.bar(np.arange(1,7), np.abs(np.array(pearson_mean))*np.sqrt(sizes), align='center')
    axes.set_xlabel('Layer')
    axes.set_ylabel('Abs(Pearson)*sqrt(N_i)')
    rects = axes.patches
    # Now make some labels
    labels = ["L%d (%d nuerons)" % (i, j) for i, j in zip(range(len(rects)), sizes)]
    plt.xticks(np.arange(1,7), labels)


def update_axes(axes, xlabel, ylabel, xlim, ylim, title, xscale, yscale, x_ticks, y_ticks, p_0, p_1
                ,font_size = 30, axis_font = 25,legend_font = 16 ):
    """adjust the axes to the ight scale/ticks and labels"""
    categories =6*['']
    labels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$']
    #The legents of the mean and the std
    leg1 = plt.legend(p_0, categories, title=r'$\|Mean\left(\nabla{W_i}\right)\|$', loc='best',fontsize = legend_font,markerfirst = False, handlelength = 5)
    leg2 = plt.legend(p_1, categories, title=r'$STD\left(\nabla{W_i}\right)$', loc='best',fontsize = legend_font ,markerfirst = False,handlelength = 5)
    leg1.get_title().set_fontsize('21')  # legend 'Title' fontsize
    leg2.get_title().set_fontsize('21')  # legend 'Title' fontsize
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)
    utils.adjustAxes(axes, axis_font=20, title_str='', x_ticks=x_ticks, y_ticks=y_ticks, x_lim=xlim, y_lim=ylim,
                     set_xlabel=True, set_ylabel=True, x_label=xlabel, y_label=ylabel, set_xlim=True, set_ylim=True,
                     set_ticks=True, label_size=font_size, set_yscale=True,
                     set_xscale=True, yscale=yscale, xscale=xscale, ytick_labels=labels, genreal_scaling=True)



def extract_array(data, name):
    results = [[data[j,k][name] for k in range(data.shape[1])] for j in range(data.shape[0])]
    return results

def update_bars_num_of_ts(num, p_ts, H_Xgt,DKL_YgX_YgT, axes, ind_array):
    axes[1].clear()
    axes[2].clear()
    axes[0].clear()
    current_pts =p_ts[num]
    current_H_Xgt = H_Xgt[num]
    current_DKL_YgX_YgT = DKL_YgX_YgT[num]
    num_of_t = [c_pts.shape[0] for c_pts in current_pts]
    x = range(len(num_of_t))
    axes[0].bar(x, num_of_t)
    axes[0].set_title('Number of Ts in every layer - Epoch number - {0}'.format(ind_array[num]))
    axes[0].set_xlabel('Layer Number')
    axes[0].set_ylabel('# of Ts')
    axes[0].set_ylim([0, 800])
    h_list, dkl_list = [], []
    for i in range(len(current_pts)):
        h_list.append(-np.dot(current_H_Xgt[i],current_pts[i]))
        dkl_list.append(np.dot(current_DKL_YgX_YgT[i].T, current_pts[i]))
    axes[1].bar(x,h_list)
    axes[2].bar(x,dkl_list)

    axes[2].bar(x, dkl_list)
    axes[1].set_title('H(X|T)', title_size = 16)
    axes[1].set_xlabel('Layer Number')
    axes[1].set_ylabel('H(X|T)')

    axes[2].set_title('DKL[p(y|x)||p(y|t)]',fontsize = 16)
    axes[2].set_xlabel('Layer Number')
    axes[2].set_ylabel('DKL[p(y|x)||p(y|t)]',fontsize = 16)

def update_bars_entropy(num, H_Xgt,DKL_YgX_YgT, axes, ind_array):
    axes[0].clear()
    current_H_Xgt =np.mean(H_Xgt[num], axis=0)
    x = range(len(current_H_Xgt))
    axes[0].bar(x, current_H_Xgt)
    axes[0].set_title('H(X|T) in every layer - Epoch number - {0}'.format(ind_array[num]))
    axes[0].set_xlabel('Layer Number')
    axes[0].set_ylabel('# of Ts')


def plot_hist(str_name, save_name='dist'):
    data_array = utils.get_data(str_name)
    params = np.squeeze(np.array(data_array['information']))
    ind_array = data_array['params']['epochsInds']
    DKL_YgX_YgT = utils.extract_array(params, 'DKL_YgX_YgT')
    p_ts = utils.extract_array(params, 'pts')
    H_Xgt = utils.extract_array(params, 'H_Xgt')

    f, (axes) = plt.subplots(3, 1)
    #axes = [axes]
    f.subplots_adjust(left=0.14, bottom=0.1, right=.928, top=0.94, wspace=0.13, hspace=0.55)
    colors = LAYERS_COLORS
    line_ani = animation.FuncAnimation(f, update_bars_num_of_ts, len(p_ts), repeat=False,
                                       interval=1, blit=False, fargs=[p_ts,H_Xgt,DKL_YgX_YgT, axes,ind_array])
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50)
    #Save the movie
    line_ani.save(save_name+'_movie.mp4',writer=writer,dpi=250)
    plt.show()

def plot_alphas(str_name, save_name='dist'):
    data_array = utils.get_data(str_name)
    params = np.squeeze(np.array(data_array['information']))
    I_XT_array = np.squeeze(np.array(extract_array(params, 'local_IXT')))
    """"
    for i in range(I_XT_array.shape[2]):
        f1, axes1 = plt.subplots(1, 1)

        axes1.plot(I_XT_array[:,:,i])
    plt.show()
    return
    """
    I_XT_array_var = np.squeeze(np.array(extract_array(params, 'IXT_vartional')))
    I_TY_array_var = np.squeeze(np.array(extract_array(params, 'ITY_vartional')))

    I_TY_array = np.squeeze(np.array(extract_array(params, 'local_ITY')))
    """
    f1, axes1 = plt.subplots(1, 1)
    #axes1.plot(I_XT_array,I_TY_array)
    f1, axes2 = plt.subplots(1, 1)

    axes1.plot(I_XT_array ,I_TY_array_var)
    axes2.plot(I_XT_array ,I_TY_array)
    f1, axes1 = plt.subplots(1, 1)
    axes1.plot(I_TY_array, I_TY_array_var)
    axes1.plot([0, 1.1], [0, 1.1], transform=axes1.transAxes)
    #axes1.set_title('Sigmma=' + str(sigmas[i]))
    axes1.set_ylim([0, 1.1])
    axes1.set_xlim([0, 1.1])
    plt.show()
    return
    """
    #for i in range()
    sigmas = np.linspace(0, 0.3, 20)

    for i in range(0,20):
        print (i, sigmas[i])
        f1, axes1 = plt.subplots(1, 1)
        axes1.plot(I_XT_array, I_XT_array_var[:,:,i], linewidth=5)
        axes1.plot([0, 15.1], [0, 15.1], transform=axes1.transAxes)
        axes1.set_title('Sigmma=' +str(sigmas[i]))
        axes1.set_ylim([0,15.1])
        axes1.set_xlim([0,15.1])
    plt.show()
    return
    epochs_s = data_array['params']['epochsInds']
    f, axes = plt.subplots(1, 1)
    #epochs_s = []
    colors = LAYERS_COLORS
    linestyles  = [ '--', '-.', '-','', ' ',':', '']
    epochs_s =[0, -1]
    for j in epochs_s:
        for i  in range(0, I_XT_array.shape[1]):

            axes.plot(sigmas, I_XT_array_var[j,i,:],color = colors[i], linestyle = linestyles[j], label='Layer-'+str(i) +' Epoch - ' +str(epochs_s[j]))
    title_str = 'I(X;T) for different layers as function of $\sigma$ (The width of the gaussian)'
    x_label = '$\sigma$'
    y_label = '$I(X;T)$'
    x_lim = [0, 3]
    utils.adjustAxes(axes, axis_font=20, title_str=title_str, x_ticks=[], y_ticks=[], x_lim=x_lim, y_lim=None,
                     set_xlabel=True, set_ylabel=True, x_label=x_label, y_label=y_label, set_xlim=True, set_ylim=False,
                     set_ticks=False,
                     label_size=20, set_yscale=False,
                     set_xscale=False, yscale=None, xscale=None, ytick_labels='', genreal_scaling=False)
    axes.legend()
    plt.show()
if __name__ == '__main__':
    #The action the you want to plot
    #Plot snapshots of all the networks
    TIME_STEMPS = 'time-stemp'
    #create movie of the networks
    MOVIE = 'movie'
    #plot networks with diffrenet number of layers
    ALL_LAYERS = 'all_layers'
    #plot networks with 5% of the data and with 80%
    COMPRAED_PERCENT = 'compare_percent'
    #plot the infomration curves for the networks with diffrenet percent of the data
    ALL_SAMPLES = 'all_samples'
    #Choose whice figure to plot
    action = COMPRAED_PERCENT
    prex = 'jobsFiles/'
    sofix = '.pickle'
    prex2 = '/Users/ravidziv/PycharmProjects/IDNNs/jobs/'
    #plot above action, the norms, the gradients and the pearson coeffs
    do_plot_action, do_plot_norms, do_plot_pearson = True, False, False
    do_plot_eig = False
    plot_movie = False
    do_plot_time_stepms = False
    #str_names = [[prex2+'fo_layerSizes=10,7,5,4,3_LastEpochsInds=9998_nRepeats=1_batch=3563_DataName=reg_1_nEpoch=10000_lr=0.0004_nEpochInds=964_samples=1_nDistSmpls=1/']]
    if action == TIME_STEMPS or action == MOVIE:
        index = 1
        name_s = prex2+ 'g_layerSizes=10,7,5,4,3_LastEpochsInds=9998_nRepeats=40_batch=3563_DataName=var_u_nEpoch=10000_lr=0.0002_nEpochInds=964_samples=1_nDistSmpls=1/'
        name_s = prex2 +'r_DataName=MNIST_sampleLen=1_layerSizes=400,200,150,60,50,40,30_lr=0.0002_nEpochInds=677_nRepeats=1_LastEpochsInds=1399_nDistSmpls=1_nEpoch=1400_batch=2544/'
        if action ==TIME_STEMPS:
            save_name = '3_time_series'
            #plot_snapshots(name_s, save_name, index)
        else:
            save_name  = 'genreal'
            plot_animation(name_s, save_name)
    else:
        if action ==ALL_LAYERS:
            mode =11
            save_name = ALL_LAYERS
            str_names = [[prex + 'ff3_5_198.pickle', prex+ 'ff3_4_198.pickle',prex + 'ff3_3_198.pickle'],[prex + 'ff3_2_198.pickle',prex + 'ff3_1_198.pickle',prex + 'ff4_1_10.pickle']]
            str_names[1][2] = prex2+'g_layerSizes=10,7,5,4,4,3_LastEpochsInds=9998_nRepeats=20_batch=3563_DataName=var_u_nEpoch=10000_lr=0.0004_nEpochInds=964_samples=1_nDistSmpls=1/'
            str_names = [[prex2 +'nbins8_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4096/',
                          prex2 +'nbins12_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4096/',
                          prex2 +'nbins18_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4096/']
                         ,[prex2 +'nbins25_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistrSmpls=1_nEpoch=10000_batch=4096/',
                           prex2 +'nbins35_DataName=var_u_sampleLen=1_layerSizes=10,7,5, 4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4096/',
                           prex2 + 'nbins50_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=964_nRepeats=5_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4096/'                         ]]
        elif action == COMPRAED_PERCENT:
            save_name = COMPRAED_PERCENT
            #mode =0
            mode = 2
            str_names    = [[prex + 'ff4_1_10.pickle', prex + 'ff3_1_198.pickle']]
        elif action == ALL_SAMPLES:
            save_name = ALL_SAMPLES
            mode =3
            str_names = [[prex+'t_32_1.pickle']]
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        str_names = [[('/').join(file_path.split('/')[:-1]) + '/']]
        if do_plot_action:
            plot_figures(str_names, mode, save_name)
        if do_plot_norms:
            plot_norms(str_names)
        if do_plot_pearson:
            plot_pearson(str_names)
        if plot_movie:
            plot_animation_each_neuron(str_names, save_name)
        if do_plot_eig:
            pass
        if do_plot_time_stepms:
            plot_snapshots(str_names[0][0], save_name, 1)

            #plot_eigs_movie(str_names)
    plt.show()


import matplotlib
matplotlib.use("TKAgg")
import numpy as np
import cPickle
from scipy.interpolate import interp1d
from numpy import linalg as LA
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import scipy.stats as sis
from scipy.interpolate import spline
from Tkinter import Tk
from tkFileDialog import askopenfilename,askdirectory
import os
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
from matplotlib.colors import colorConverter
import matplotlib.animation as animation
import cStringIO
from PIL import Image
import math

def plotByEpoch(I_XT_array, I_TY_array,axes,epochsInds, f, index_i, index_j, size_ind,font_size, y_ticks, x_ticks,colorbar_axis,title_str, axis_font, bar_font, save_name):
    if size_ind!=-1:
        indexs_range = size_ind
    else:
        indexs_range = I_XT_array.shape[3]-1
    cmap = plt.get_cmap('gnuplot')
    #print epochsInds.shape, indexs_range
    colors = [cmap(i) for i in np.linspace(0, 1, epochsInds[indexs_range]+1)]
    nums_arc= -1
    #colors[400] = 'g'
    for index_in_range in range(0, indexs_range):
        XT, TY = [], []
        for layer_index in range(0, I_XT_array.shape[4]):
                XT.append(np.mean(I_XT_array[:, -1, nums_arc, index_in_range, layer_index], axis=0))
                TY.append(np.mean(I_TY_array[:, -1, nums_arc, index_in_range, layer_index], axis=0))
        if epochsInds[index_in_range] ==160:
            axes[index_i, index_j].plot(XT, TY, marker='o', linestyle='-', markersize=19, markeredgewidth=0.04,
                                        linewidth=2.1,
                                        color='g',zorder=10)
        else:
            axes[index_i, index_j].plot(XT, TY, marker='o', linestyle='-', markersize=12, markeredgewidth=0.01, linewidth=0.2,
                            color=colors[epochsInds[index_in_range]])
    if index_i ==axes.shape[0]-1:
        axes[index_i, index_j].set_xlabel('$I(X;T)$', fontsize=font_size)

    axes[index_i, index_j].set_xlim([0,12.2])
    axes[index_i, index_j].set_ylim([0, 1.02])
    axes[index_i, index_j].set_title(title_str, fontsize=axis_font+2)
    #axes[index].set_yticks(y_ticks)
    axes[index_i, index_j].tick_params(axis='y', labelsize=axis_font)
    axes[index_i, index_j].tick_params(axis='x', labelsize =axis_font)

    axes[index_i, index_j].set_xticks(x_ticks)
    axes[index_i, index_j].set_yticks(y_ticks)
    x = np.arange(0, 13)
    y = .045 * x + 0.44
    #axes[index_i, index_j].plot(x, y, linewidth = 2,linestyle='-', color = 'g')
    if index_j ==0:
        axes[index_i, index_j].set_ylabel('$I(T;Y)$', fontsize=font_size)
    if index_i ==axes.shape[0]-1 and index_j ==axes.shape[1]-1:
        #s = plt.figure()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar_ax = f.add_axes(colorbar_axis)
        #cbar = f.colorbar(sm, ticks=[0, .98], cax=cbar_ax)
        #cbar.ax.set_yticklabels(['0', "$10^4$"],fontsize=bar_font)
        #cbar.ax.set_title('Epochs', fontsize=bar_font)

        cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=bar_font)
        cbar.set_label('Epochs', size=bar_font)

        # cbar.ax.set_title('Data', fontsize=)
        cbar.ax.text(0.5, -0.01, '0', transform=cbar.ax.transAxes,
                     va='top', ha='center', size=bar_font)
        cbar.ax.text(0.5, 1.0, '$10^4$', transform=cbar.ax.transAxes,
                     va='bottom', ha='center', size=bar_font)

        #s.savefig('s.pdf')

        f.savefig(save_name+'.JPG', dpi=150, format='JPG')





def plotLByLayer(I_XT_array, I_TY_array,axes,epochsInds, f,colorbar_axis,name_to_save,font_size, axis_font, bar_font):

    indexs_range = I_XT_array.shape[2]-1
    cmap = plt.get_cmap('gnuplot')
    #print epochsInds
    colors = [cmap(i) for i in np.linspace(0, 1, indexs_range+1)]
    print I_XT_array.shape
    for index_in_range in range(0, indexs_range):
        XT, TY = [], []
        #print index_in_range
        for layer_index in range(0, I_XT_array.shape[4]):
            XT.append(np.mean(I_XT_array[:, -1, index_in_range, -1, layer_index], axis=0))
            TY.append(np.mean(I_TY_array[:, -1, index_in_range, -1, layer_index], axis=0))
        axes[0][0].plot( XT, marker='o', linestyle='-', linewidth=1,markersize=13, markeredgewidth=0.2,
                        color=colors[index_in_range])
        axes[0][1].plot(TY, marker='o', linestyle='-', linewidth=1, markersize=13, markeredgewidth=0.2,
                        color=colors[index_in_range])

    axes[0][0].tick_params(axis='y', labelsize=axis_font)
    axes[0][1].tick_params(axis='y', labelsize=axis_font)
    axes[0][0].tick_params(axis='x', labelsize=axis_font)
    axes[0][1].tick_params(axis='x', labelsize=axis_font)
    #axes[0][0].set_xlabel('Layers', fontsize=font_size)
    #axes[0][1].set_xlabel('Layers', fontsize=font_size)
    axes[0][0].set_ylabel('$I(X;T)$', fontsize=font_size)
    axes[0][1].set_ylabel('$I(T;Y)$', fontsize=font_size)
    axes[0][0].set_xlabel('Layer', fontsize=font_size)
    axes[0][1].set_xlabel('Layer', fontsize=font_size)

    axes[0][0].set_xlim([0, 5.1])
    axes[0][1].set_xlim([0, 5.1])

    axes[0][0].set_ylim([1, 12.15])
    axes[0][1].set_ylim([0.36, 1.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []

    cbar_ax = f.add_axes(colorbar_axis)
    cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label('Training Data', size=bar_font)

    #cbar.ax.set_title('Data', fontsize=)
    cbar.ax.text(0.5, -0.01, '3%', transform=cbar.ax.transAxes,
               va='top', ha='center', size= bar_font)
    cbar.ax.text(0.5, 1.0, '85%', transform=cbar.ax.transAxes,
               va='bottom', ha='center', size=bar_font)
    #cbar.ax.set_yticklabels(['3%', "85%"], size=100)
    #f.savefig('xy_layers.png', dpi=300)
    f.savefig('xy_layers' + '.JPG', dpi=150, format='JPG')

def plotBySamples(I_XT_array, I_TY_array,axes,epochsInds, f, index_i, index_j, size_ind,font_size, y_ticks, x_ticks,colorbar_axis,title_str, axis_font, bar_font,save_name ):
    if size_ind!=-1:
        indexs_range = size_ind
    else:
        indexs_range = I_XT_array.shape[2]-1
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, indexs_range+1)]
    nums_epoch= -1
    print (I_XT_array.shape)
    for index_in_range in range(0, indexs_range):
        XT, TY = [], []
        #print (I_XT_array.shape)
        for layer_index in range(0, I_XT_array.shape[4]):
                XT.append(np.mean(I_XT_array[:, -1, index_in_range, nums_epoch, layer_index], axis=0))
                TY.append(np.mean(I_TY_array[:, -1, index_in_range,nums_epoch, layer_index], axis=0))
        #print I_XT_array
        axes[index_i, index_j].plot(XT, TY, marker='o', linestyle='-', markersize=12, markeredgewidth=0.2, linewidth=0.5,
                         color=colors[index_in_range])
    if index_i == axes.shape[0] - 1:
        axes[index_i, index_j].set_xlabel('$I(X;T)$', fontsize=font_size)

    axes[index_i, index_j].set_xlim([1, 12.2])
    axes[index_i, index_j].set_ylim([0.3, 1.02])
    axes[0][0].tick_params(axis='y', labelsize=axis_font)
    axes[0][0].tick_params(axis='x', labelsize=axis_font)

    #axes[index_i, index_j].set_title(title_str, fontsize=font_size)
    # axes[index].set_yticks(y_ticks)
    #axes[index_i, index_j].tick_params(axis='y', labelsize=font_size)
    axes[index_i, index_j].set_xticks(x_ticks)
    axes[index_i, index_j].set_yticks(y_ticks)

    #axes[index_i, index_j].tick_params(axis='x', labelsize=font_size)
    if index_j == 0:
        axes[index_i, index_j].set_ylabel('$I(T;Y)$', fontsize=font_size)
    if index_i == axes.shape[0] - 1 and index_j == axes.shape[1] - 1:
        # s = plt.figure()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []

        cbar_ax = f.add_axes(colorbar_axis)
        cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=bar_font)
        cbar.set_label('Training Data', size=bar_font)

        # cbar.ax.set_title('Data', fontsize=)
        cbar.ax.text(0.5, -0.01, '3%', transform=cbar.ax.transAxes,
                     va='top', ha='center', size=bar_font)
        cbar.ax.text(0.5, 1.0, '85%', transform=cbar.ax.transAxes,
                     va='bottom', ha='center', size=bar_font)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar_ax = f.add_axes(colorbar_axis)
        #cbar = f.colorbar(sm, ticks=[0, .98], cax=cbar_ax)
        #cbar.ax.set_yticklabels(['3%', "85%"], fontsize=font_size - 2)
        #cbar.ax.set_title('Data', fontsize=font_size - 2)
        # s.savefig('s.pdf')
        #f.savefig('all_samples_data.png', dpi=300)
        f.savefig('all_samples_data' + '.JPG', dpi=150, format='JPG')


def calcVelocity(data, epochs):
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
def update_line(num, data, axes,epochsInds,information_each_neuron):
    print num
    #if num not in epochsInds:
    #    return
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()

    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange']
    l1, l2, l3, l4,l5,l6 =[],[],[],[],[],[]
    ls = [l1, l2, l3, l4, l5, l6]
    #print (len(information_each_neuron), len(information_each_neuron[0]),len(information_each_neuron[0][0]))
    for k in range(len(information_each_neuron)):
        for j in range(len(information_each_neuron[k])):
            for i in range(len(information_each_neuron[k][j])):
                per_neuron_information = information_each_neuron[k][j][i]
                #print ('i - ',k , j, i)
                for layer in range(per_neuron_information.shape[1]):
                    #print ('layer - ', layer)
                    per_neuron_information_layer =per_neuron_information[num, layer,:]
                    ls[layer].append(per_neuron_information_layer)
    sumsX ,sumsY= [],[]
    for layer in range(len(ls)):
        per_neuron_information_layer = np.array(ls[layer])
        c_sumY = np.mean(np.sum(np.vstack(per_neuron_information_layer[:,1]), axis=1), axis=0)
        c_sumX = np.mean(np.sum(np.vstack(per_neuron_information_layer[:,0]), axis=1), axis=0)

        #print c_sumY, c_sumX
        sumsX.append(c_sumX)
        sumsY.append(c_sumY)
        #print (per_neuron_information_layer.shape)
        axes[0].scatter(np.mean(per_neuron_information_layer[:,0], axis=0),
                    np.mean(per_neuron_information_layer[:,1], axis=0), color=colors[layer], s=35,
                    edgecolors='black', alpha=0.85)
    # axes[k].scatter(np.mean(data[0, :, k, sample_num, num, :], axis=0), np.mean(data[1, :, k, sample_num, num, :] ,axis=0), color = colors, s = 35,edgecolors = 'black',alpha = 0.85)
    x_sum = np.array(sumsX)
    y_sum = np.array(sumsY)

    x_data = np.mean(data[0, :, -1, -1, num, :], axis=0)
    y_data = np.mean(data[1, :, -1, -1, num, :], axis=0)
    #print (x_data-x_sum).shape, (y_data-y_sum).shape
    axes[1].scatter(x_data, y_data,  color = colors, s = 35,edgecolors = 'black',alpha = 0.85)
    axes[2].scatter(x_sum, y_sum,  color = colors, s = 45,edgecolors = 'black',alpha = 0.85)
    #print x_sum, y_sum
    #axes[2].scatter(np.abs(x_data-x_sum) /(x_data+x_sum),np.abs(y_data-y_sum) /(y_data+y_sum), color = colors, s = 35,edgecolors = 'black',alpha = 0.85)


    axes[0].set_title('Number of epochs - ' + str(epochsInds[num]))
    axes[0].set_xlim(0, 12.2)
    axes[0].set_ylim(0.0, 1)
    axes[1].set_xlim(0, 12.2)
    axes[1].set_ylim(0.0, 10.5)
    axes[2].set_xlim(0, 55)
    axes[2].set_ylim(0.0, 5)
    return
    for k in range(0,len(axes)):
        segs = []
        for i in range(0, data.shape[1]):
            x = data[0, i, k, -1, num, :]
            y = data[1, i, k, -1, num, :]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]],
                                      axis=1)
            segs.append(segments)
        segs = np.array(segs).reshape(-1, 2, 2)
        axes[k].clear()
        #print (segments.shape)
        cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'pink', 'orange'])

        lc = LineCollection(segs, cmap=cmap, linestyles='solid',linewidths = 0.3, alpha = 0.6)
        #axes[k].add_collection(lc)
        lc.set_array(np.arange(0,5))
        sample_num = 0
        colors  = ['red', 'green', 'blue', 'yellow', 'pink', 'orange']
        cmap1 = plt.get_cmap('gnuplot')
        colors = [cmap1(i) for i in np.linspace(0, 1, data.shape[3])]
        for i in range(len(information_each_neuron)):
            for j in range(len(information_each_neuron[k])):
                for k in range(information_each_neuron[k][j]):
                    per_neuron_information = information_each_neuron[k][j][i]

        for sample_num in range(data.shape[3]):
            axes[k].scatter(np.mean(information_each_neuron[0, :, k, sample_num, num, :], axis=0), np.mean(data[1, :, k, sample_num, num, :] ,axis=0), color = colors, s = 35,edgecolors = 'black',alpha = 0.85)

            #axes[k].scatter(np.mean(data[0, :, k, sample_num, num, :], axis=0), np.mean(data[1, :, k, sample_num, num, :] ,axis=0), color = colors, s = 35,edgecolors = 'black',alpha = 0.85)
            #axes[k].scatter(data[0, :, k, sample_num, num, :],data[1, :, k, sample_num, num, :], color=colors[sample_num], s=35, edgecolors='black',
            #            alpha=0.3)
        axes[k].set_xlim(0,12.2)
        axes[k].set_ylim(0.0, 1)
    axes[0].set_title('Number of epochs - ' + str(epochsInds[num]))
    """"
    fig1.suptitle('Number of epochs - ' + str(epochs[num]), fontsize=14)

    ax2.plot(epochs[:num], 1-np.mean(train_data[:,0,0,:num],axis=0),color = 'r')
    ax2.plot(epochs[:num], 1-np.mean(test_data[:,0,0,:num], axis=0), color='g')
    nereast_val = np.searchsorted(epochs_bins, epochs[num], side='right')
    ax2.set_xlim([0,epochs_bins[nereast_val]])
    ax2.set_ylim([0, 0.5])
    ax2.legend(('Train', 'Test'), loc=1)
    """
def update_lineAllThree(nums,print_loss, data, axes,epochsInds,train_data,test_data,epochs_bins, loss_train_data, loss_test_data, to_do, font_size,axis_font):
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange']
    for i in range(len(nums)):
        num = nums[i]
        print num
        print data.shape
        for layer_num in range(data.shape[3]):
            axes[i].scatter(data[0, :, num, layer_num], data[1, :, num, layer_num], color = colors[layer_num], s = 105,edgecolors = 'black',alpha = 0.85)
        axes[i].set_xlim([0, 12.2])
        axes[i].set_ylim([0, 1.03])
        x_ticks = [0,2, 4, 6, 8, 10]
        axes[i].set_xticks(x_ticks)
        axes[i].tick_params(axis='x', labelsize=axis_font)
        if to_do[i][0]:
            axes[i].set_xlabel('$I(X;T)$', fontsize=font_size)
            axes[i].tick_params(axis='x', labelsize=axis_font)

        else:
            x_ticks = [0, 4,8, 12]
            axes[i].set_xticks(x_ticks)
            axes[i].tick_params(axis='x', labelsize=axis_font)
        # axes[index].set_yticks(y_ticks)
        if to_do[i][1]:
            axes[i].set_ylabel('$I(T;Y)$', fontsize=font_size)
            axes[i].tick_params(axis='y',labelsize=axis_font)
            #y_ticks = []
            #axes[i].set_yticks(y_ticks)
            #axes[i].tick_params(axis='y', which='both', labelbottom='off', top='off', bottom='off')
        else:
            y_ticks = []
            #axes[i].set_yticks(y_ticks)
            #axes[i].tick_params(axis='y', which='both', labelbottom='off', top='off',bottom='off')

def update_lineAll(num,print_loss, data, axes,epochsInds,train_data,test_data,epochs_bins, loss_train_data, loss_test_data):
    print num
    font_size = 18
    axis_font = 16
    cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'pink', 'orange'])
    cmap1 = plt.get_cmap('gnuplot')
    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange']
    segs = []
    #print data.shape
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
    #print data.shape, len(colors)
    for layer_num in range(data.shape[3]):
        axes[0].scatter(data[0, :, num, layer_num], data[1, :, num, layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)
    axes[0].set_xlim(0,12.2)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_title('Information Plane - Epoch number - ' + str(epochsInds[num]), fontsize = axis_font)
    axes[0].set_ylabel('$I(T;Y)$', fontsize=font_size)
    axes[0].set_xlabel('$I(X;T)$', fontsize=font_size)
    axes[1].set_ylabel('Precision', fontsize=font_size)
    axes[1].set_xlabel('# Epochs', fontsize=font_size)
    axes[0].tick_params(axis='x', labelsize=axis_font)
    axes[0].tick_params(axis='y', labelsize=axis_font)
    axes[1].tick_params(axis='x', labelsize=axis_font)
    axes[1].tick_params(axis='y', labelsize=axis_font)
    axes[1].set_title('Precision as function of the epochs', fontsize = axis_font)

    #axes[1].plot(epochsInds[:num], 1-np.mean(train_data[:,:num],axis=0),color = 'r')
    if len(axes)>1:
        axes[1].plot(epochsInds[:num], 1-np.mean(test_data[:,:num], axis=0), color='g')
        #axes[1].plot(epochsInds[:num], np.mean(loss_train_data[:, :num], axis=0), color='b')
        if print_loss:
            axes[1].plot(epochsInds[:num], np.mean(loss_test_data[:, :num], axis=0), color='y')
        nereast_val = np.searchsorted(epochs_bins, epochsInds[num], side='right')
        axes[1].set_xlim([0,epochs_bins[nereast_val]])
        axes[1].set_ylim([0, 0.9])
        axes[1].legend(('Accuracy', 'Loss Function'), loc='best')

def loadPoints(name):
    with open(name + '.mat', 'rb') as handle:
        d = sio.loadmat(name + '.mat')
        Ix_c = d['Ix_c']
        Iy_c = d['Iy_c']
        bc = d['bc']
    return np.squeeze(Ix_c), np.squeeze(Iy_c), np.squeeze(bc)
def loadDAnnealing(name,max_beta = 300, min_beta=0.8, dt = 0.1):
    with open(name + '.mat', 'rb') as handle:
        d = sio.loadmat(name + '.mat')
        F = d['F']
        ys = d['y']
        PXs = np.ones(len(F)) / len(F)
        f_PYs = np.mean(ys)
    PYs = np.array([f_PYs, 1 - f_PYs])
    PYX = np.concatenate((np.array(ys)[None, :], 1 - np.array(ys)[None, :]))
    mybetaS = 2 ** np.arange(np.log2(min_beta), np.log2(max_beta), dt)
    mybetaS = mybetaS[::-1]
    PTX0 = np.eye(PXs.shape[0])
    return mybetaS, np.squeeze(PTX0), np.squeeze(PXs), np.squeeze(PYX), np.squeeze(PYs)
def getData(total_name):
    import os.path
    if os.path.isfile(total_name + 'data.pickle'):
        curent_f = open(total_name + 'data.pickle', 'rb')
        #curent_f = open(total_name, 'rb')
        d2 = cPickle.load(curent_f)
        #print d2['params']
        if False:
            epochs = d2['params']['epochsInds']
        else:
            epochs = [20]
        I_XT_array2, I_TY_array2 = 0,0
        #if 'information' in d2:
        I_XT_array2 = d2['information'][:, :, :, :, :, 0]
        I_TY_array2 = d2['information'][:, :, :, :, :, 1]
        train_data = d2['train_error']
        test_data = d2['test_error']
        ws ,loss_train_data ,loss_test_data=0,0,0
        information_each_neuron,norms1,norms2,gradients =0,0,0, 0
        print d2.keys()
        if 'ws_all' in d2:
            ws = d2['ws_all']
        if 'information_each_neuron' in d2:
            information_each_neuron = d2['information_each_neuron']
        data = np.array([I_XT_array2, I_TY_array2])
        if 'loss_train' in d2:
            loss_train_data =d2['loss_train']
            loss_test_data = d2['loss_test']
        params = d2['params']
        if 'var_grad_val' in d2:
            gradients = d2['var_grad_val']
        if 'l1_norms' in d2:
            norms1 = d2['l1_norms']
            norms2 = d2['l2_norms']
        epochsInds = (params['epochsInds']).astype(np.int)
        #epochsInds = np.arange(0, data.shape[4])

        normalization_factor = 1
        print (len(epochsInds))
        #normalization_factor = 1
    else:
        curent_f = open(total_name, 'rb')
        d2 = cPickle.load(curent_f)
        data1 = d2[0]
        data =  np.array([data1[:, :, :, :, :, 0], data1[:, :, :, :, :, 1]])
        epochs, train_data, test_data, ws,loss_train_data,loss_test_data, norms1, norms2,gradients =0,0,0,0,0,0, 0,0,0
        params = d2[-1]
        epochsInds = np.arange(0, data.shape[4])
        normalization_factor =1/np.log2(2.718281)
        information_each_neuron = 0
        #epochsInds = np.round(2 ** np.arange(np.log2(1), np.log2(10000), .01)).astype(np.int)
    return data,epochs,train_data,test_data,ws,params,epochsInds,normalization_factor,information_each_neuron,loss_train_data,loss_test_data, \
           norms1, norms2,gradients
def updateLine1():

    print ('Here')
def calcInformation(p_yhat_x):
    #return 0,0
    min_beta,max_beta,dt = 1 , 600, 0.1
    mybetaS = 2 ** np.arange(np.log2(min_beta), np.log2(max_beta), dt)
    mybetaS = mybetaS[::-1]
    PXs = np.ones(4096) / 4096
    PTX0 = np.eye(PXs.shape[0])
    Pyhatx = np.array([c_p_yhat_x * PXs for c_p_yhat_x in p_yhat_x])
    PYs = np.sum(Pyhatx, axis=1)
    Hy = np.nansum(-PYs * np.log2(PYs))
    Hyx = - np.nansum(np.dot(np.multiply(p_yhat_x, np.log2(p_yhat_x)), PXs))
    IXY = Hy - Hyx
    print (IXY)
    #return 0, 0
    local_ICX, local_IYC = revereseAnnealing.main_from_source(mybetaS, PTX0, PXs, p_yhat_x, PYs, ITER=5)
    return local_ICX, local_IYC
def plotAnimationAll(name_s,save_name):
    prex2 = 'jobs/'
    print_loss = False
    font_size = 45
    name_s  = prex2 +name_s
    print_loss  = False
    #name_s = prex2 +'r_layersSizes=[[10, 7, 5, 4, 3]]_LastEpochsInds=9998_numRepeats=50_batch=3563_DataName=g1_numEphocs=10000_learningRate=7e-05_numEpochsInds=964_samples=1_num_of_disribuation_samples=1/'
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, norms2, gradients = getData(name_s)
    f, (axes) = plt.subplots(2, 1)
    f.subplots_adjust(left=0.14, bottom=0.1, right=.928, top=0.94, wspace=0.13, hspace=0.55)
    axes[0].set_ylabel('$I(T;Y)$', fontsize=font_size)
    axes[0].set_title('Information Plane')
    axes[0].set_xlabel('$I(X;T)$', fontsize=font_size)
    axes[1].set_ylabel('Precision', fontsize=font_size)
    axes[1].set_xlabel('# Epochs', fontsize=font_size)
    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]
    new_x = np.arange(0,epochsInds[-1])
    Ix = np.squeeze(data[0,:,-1,-1, :, :])
    Iy = np.squeeze(data[1,:,-1,-1, :, :])
    interp_data_x = interp1d(epochsInds,  Ix, axis=1)
    interp_data_y = interp1d(epochsInds,  Iy, axis=1)
    train_data = interp1d(epochsInds,  np.squeeze(train_data), axis=1)(new_x)
    test_data = interp1d(epochsInds,  np.squeeze(test_data), axis=1)(new_x)
    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]
    if print_loss:
        loss_train_data =  interp1d(epochsInds,  np.squeeze(loss_train_data), axis=1)(new_x)
        loss_test_data=interp1d(epochsInds,  np.squeeze(loss_test_data), axis=1)(new_x)
    new_data  = np.array([interp_data_x(new_x), interp_data_y(new_x)])
    line_ani = animation.FuncAnimation(f, update_lineAll, len(new_x),repeat=False,
                                       interval=1, blit=False, fargs=(print_loss, new_data, axes,new_x,train_data,test_data,epochs_bins, loss_train_data,loss_test_data))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100)
    line_ani.save(save_name+'_movie.mp4',writer=writer,dpi=250)
    plt.show()
def printTimeFigures(name_s,save_name, i):
    fig_size = (14, 10)
    font_size = 64
    axis_font = 49
    bar_font = 25
    #fig_size = (14, 14)

    time_stemps = [20, 500,700, 963]

    if True:
        font_size = 36
        axis_font = 28
        fig_size = (14, 6)

        f, (axes) = plt.subplots(1, 3, sharey=True, figsize=fig_size)

        #axes = np.vstack(np.array([axes]))
        print axes.shape
        f.subplots_adjust(left=0.095, bottom=0.18, right=.99, top=0.97, wspace=0.03, hspace=0.03)
        time_stemps = [13, 180, 963]
        to_do = [[True, True], [True, False], [True, False]]

    else:
        fig_size = (14, 10)
        f, (axes) = plt.subplots(1, 1, sharey=True, figsize=fig_size)

        axes = [axes]

        if i==0 or i==2:
            f.subplots_adjust(left=0.15, bottom=0.16, right=.985, top=0.98, wspace=0.03, hspace=0.03)
        else:
            f.subplots_adjust(left=0.05, bottom=0.16, right=.895, top=0.98, wspace=0.03, hspace=0.03)
        time_stemps = [time_stemps[i]]
        to_dos = [[False, True], [False, False], [True, True], [True, False]]
        to_do = to_dos[i]
    prex2 = 'jobs/'
    print_loss = False

    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]
    name_s = prex2 + name_s
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, norms2, gradients = getData(
    name_s)
    data = np.squeeze(data)
    update_lineAllThree(time_stemps, print_loss, data, axes, epochsInds, train_data, test_data, epochs_bins, loss_train_data,
                   loss_test_data, to_do,font_size,axis_font)
    f.savefig(save_name +'.png' ,dpi=500)
    f.savefig(save_name + '.jpg', dpi=200, format='jpg')

def plotFigures(str_names, mode, save_name):
    yticks = [0,  0.2, 0.4, 0.6, 0.8, 1]
    xticks = [1,3,5,7,9,11]
    #two figures with error bar
    if mode==0:
        font_size = 34
        axis_font = 28
        bar_font = 28
        fig_size = (14, 6.5)
        title_strs = [['','']]
        f, (axes) = plt.subplots(1, 2, sharey=True, figsize=fig_size)
        sizes = [[-1, 1400]]
        colorbar_axis = [0.935, 0.14, 0.027, 0.75]
        axes = np.vstack(axes).T
        f.subplots_adjust(left=0.09, bottom=0.15, right=.928, top=0.94, wspace=0.03, hspace=0.04)
    #for 3 rows with 2 colmus
    if mode==1:
        font_size = 34
        axis_font = 26
        bar_font = 28
        fig_size = (14,19)
        xticks = [1, 3, 5, 7, 9, 11]
        yticks = [0, 0.2, 0.4, 0.6, 0.8,1]
        title_strs = [['One hidden layer', 'Two hidden layers'], ['Three hidden layers', 'Four hidden layers'],
                      ['Five hidden layers', 'Six hidden layers']]
        f, (axes) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=fig_size)
        f.subplots_adjust(left=0.09, bottom=0.08, right=.92, top=0.94, wspace=0.03, hspace=0.15)
        colorbar_axis = [0.93, 0.1, 0.035, 0.76]
        sizes = [[1010, 1010], [1017,1020], [1700, 920]]
    #on3 figure
    if mode==2:
        axis_font = 28
        bar_font = 28
        fig_size = (14, 10)
        font_size = 34
        f, (axes) = plt.subplots(1, len(str_names), sharey=True, figsize=fig_size)
        if len(str_names) == 1:
            axes = np.vstack(np.array([axes]))
        f.subplots_adjust(left=0.084, bottom=0.12, right=.87, top=0.99, wspace=0.03, hspace=0.03)
        colorbar_axis = [0.905, 0.12, 0.03, 0.82]
        xticks = [1, 3,5,7, 9, 11]
        yticks = [0.3, 0.5, 0.7, 0.9]
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
            yticks = [0.3, 0.4, 0.6, 0.8, 1 ]
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
        f.subplots_adjust(left=0.061, bottom=0.15, right=.933, top=0.94, wspace=0.17, hspace=0.04)
    for i in range(len(str_names)):
        for j in range(len(str_names[i])):
            name_s = str_names[i][j]
            data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data,norms1, norms2, gradients = getData(name_s)
            #convert between log2 and log e
            I_XT_array = data[0, :, :, :, :, :] / normalize_factor
            I_TY_array = data[1, :, :, :, :, :] / normalize_factor
            if mode ==3:
                plotBySamples(I_XT_array, I_TY_array, axes,epochsInds,f, i, j,sizes[i][j],font_size,yticks,xticks,colorbar_axis,title_strs[i][j],axis_font, bar_font ,save_name)
            else:
                plotByEpoch(I_XT_array, I_TY_array, axes, epochsInds, f, i, j, sizes[i][j], font_size, yticks, xticks,
                            colorbar_axis, title_strs[i][j], axis_font, bar_font, save_name)
            #plotLByLayer(I_XT_array, I_TY_array, axes, epochsInds, f, colorbar_axis, name,font_size, axis_font, bar_font)

def plotNorms(str_names):
    f, (axes) = plt.subplots(1,1)
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, norms2, gradients = getData(str_names)
    axes.plot(epochsInds, np.mean(norms1[:,0,0,:], axis=0), color='g')
    axes.plot(epochsInds, np.mean(norms2[:,0,0,:], axis=0), color='b')
    axes.legend(('L1 norm', 'L2 norm'))
    axes.set_xlabel('Epochs')

def plotLayers(name_s,name_to_save):
    colorbar_axis = [0.93, 0.1, 0.02, 0.79]
    figsize =    (14, 10)
    f, (axes) = plt.subplots(1, 2, figsize = figsize)
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, \
    loss_test_data, norms1, norms2, gradients = getData(
        name_s)
    I_XT_array = data[0, :, :, :, :, :] / normalize_factor
    I_TY_array = data[1, :, :, :, :, :] / normalize_factor
    plotLByLayer(I_XT_array, I_TY_array,axes,epochsInds, f,colorbar_axis,name_to_save)

# calculates the mean
def mean(x):
    sum = 0.0
    for i in x:
         sum += i
    return sum / len(x)

# calculates the sample standard deviation
def sampleStandardDeviation(x):
    sumv = 0.0
    for i in x:
         sumv += (i)**2
    return math.sqrt(sumv/(len(x)-1))

# calculates the PCC using both the 2 functions above
def pearson(x,y):
    scorex = []
    scorey = []

    for i in x:
        scorex.append((i)/sampleStandardDeviation(x))

    for j in y:
        scorey.append((j)/sampleStandardDeviation(y))

# multiplies both lists together into 1 list (hence zip) and sums the whole list
    return (sum([i*j for i,j in zip(scorex,scorey)]))/(len(x)-1)

def calcvVarNeurons(name_s):
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, norms2, gradients = getData(
        name_s)
    f = plt.figure(figsize=(12, 8))
    axes = f.add_subplot(111)
    lists_mean =[]
    sizes =[10,7, 5, 4,3,2 ]

    for layer_index in range(6):
        layer = layer_index
        lists_mean_t =[]
        for k in range(len(ws)):
            ws_current = np.squeeze(ws[k][0][0][-1])

            person_t = []
            for neuron in range(len(ws_current[layer])):
                current_vec_first = ws_current[layer][neuron]

                for neuron_second in range(neuron+1, len(ws_current[layer])):
                    current_vec = ws_current[layer][neuron_second]
                    if neuron ==0 and neuron_second ==0:
                        print layer_index, len(current_vec)
                    #current_vec_first =np.mean(current_vec_first)
                    #current_vec =np.mean(current_vec)
                    #print (np.mean(current_vec))
                    pearson_c, p_val =sis.pearsonr(current_vec_first, current_vec)
                    #pearson_c = pearson(current_vec_first, current_vec)
                    #print p_2, pearson_c
                    person_t.append(pearson_c)
            lists_mean_t.append(np.mean(person_t))
        lists_mean.append(np.mean(lists_mean_t))
        # for i in range(len(lists)):
    axes.bar(np.arange(1,7), np.abs(np.array(lists_mean))*np.sqrt(sizes), align='center')
    axes.set_xlabel('Layer')
    axes.set_ylabel('Abs(Pearson)*sqrt(N_i)')
    rects = axes.patches

    # Now make some labels
    labels = ["L%d (%d nuerons)" % (i,j) for i,j in zip(xrange(len(rects)), sizes)]
    plt.xticks(np.arange(1,7), labels)

    for rect, label in zip(rects, labels):
        print label
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2, 2, label, ha='center', va='bottom')

    #strs.append('L' + str(layer_index + 1) + ' Person Coeff')


    #updateAxes(axes, strs, '# Epochs', 'Mean/Std', [0, 9000], [0, 0.16], '', 'linear', 'linear',
    #           [0, 2000, 4000, 6000, 8000], [0, 0.04, 0.08, 0.12, 0.16])

def updateAxes(axes, xlabel,ylabel,  xlim, ylim, title, xscale, yscale, x_ticks, y_ticks, p_0, p_1):
    font_size = 30
    axis_font = 25
    legend_font = 16
    categories =6*['']
    #axes.axvline(x=370, color='grey', linestyle=':', linewidth = 4)
    leg1 = plt.legend([p_0[0],p_0[1],p_0[2],p_0[3],p_0[4], p_0[5]], categories, title=r'$\|Mean\left(\nabla{W_i}\right)\|$', loc='best',fontsize = legend_font,markerfirst = False, handlelength = 5)
    leg2 = plt.legend([p_1[0],p_1[1],p_1[2],p_1[3],p_1[4], p_1[5]], categories, title=r'$STD\left(\nabla{W_i}\right)$', loc='best',fontsize = legend_font ,markerfirst = False,handlelength = 5)
    leg1.get_title().set_fontsize('21')  # legend 'Title' fontsize
    leg2.get_title().set_fontsize('21')  # legend 'Title' fontsize

    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)

    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlabel(xlabel,fontsize=font_size)
    axes.set_ylabel(ylabel, fontsize=font_size)

    axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    labels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$']

    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks )
    axes.tick_params(axis='x', labelsize=axis_font)
    axes.tick_params(axis='y', labelsize=axis_font)
    #axes.ticklabel_format(axis='y', style='sci')
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.xaxis.major.formatter._useMathText = True
    axes.set_yticklabels(labels)
    axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    #axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def updateAxesNorms(axes, xlabel, ylabel):
    font_size = 30
    axis_font = 25
    legend_font = 16
    categories = [r'$\|W_1\|$', r'$\|W_2\|$', r'$\|W_3\|$', r'$\|W_4\|$', r'$\|W_5\|$', r'$\|W_6\|$']
    axes.axvline(x=370, color='grey', linestyle=':', linewidth=4)
    axes.legend(categories , loc='best', fontsize=legend_font)

    axes.set_xlabel(xlabel, fontsize=font_size)
    axes.set_ylabel(ylabel, fontsize=font_size)
    #axes.set_xticks(x_ticks)
    #axes.set_yticks(y_ticks)
    axes.tick_params(axis='x', labelsize=axis_font)
    axes.tick_params(axis='y', labelsize=axis_font)

def plotGrads(name_s):
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, norms2, gradients = getData(
        name_s)
    t_lists, t_list_mean = [],[]
    fig_size = (14, 10)
    f_norms, (axes_norms) = plt.subplots(1, 1, figsize=fig_size)
    f_log, (axes_log) = plt.subplots(1, 1,figsize=fig_size)
    f_log.subplots_adjust(left=0.097, bottom=0.11, right=.95, top=0.95, wspace=0.03, hspace=0.03)
    p_1, p_0 =[],[]
    colors = ['red','c', 'blue', 'green', 'orange', 'purple']
    l2_norm_total = []
    for layer_index in range(6):
        layer = layer_index*2
        print layer_index
        for k in range(len(gradients)):
            grad = np.squeeze(gradients[k][0][0])
            ws_in = np.squeeze(ws[k][0][0])
            lists ,lists_mean,lists_mean,l2_norm= [], [] ,[],[]
            for epoch_number in range(len(grad)):
                if type(ws_in[epoch_number][layer_index]) is list:
                    flatted_list = [item for sublist in ws_in[epoch_number][layer_index] for item in sublist]
                else:
                    flatted_list = ws_in[epoch_number][layer_index]
                l2_norm.append(LA.norm(np.array(flatted_list), ord=2))
                c_mean,c_var,total_w = [], [],[]
                #current_list = np.array([grad[epoch_number][i][layer][neuron][epoch] for i in range(len(grad[epoch_number]))  for neuron in range(len(grad[epoch_number][0][layer])) for epoch in range(len(grad[epoch_number][0][layer][0])) ])
                #current_list = np.array([[grad[epoch_number][i][layer][neuron] for i in range(len(grad[epoch_number]))]  for neuron in range(len(grad[epoch_number][0][layer])) ])
                current_list = []
                for i in range(len(grad[epoch_number])):
                    current_list_inner = []
                    for neuron in range(len(grad[epoch_number][0][layer])):
                        c_n = grad[epoch_number][i][layer][neuron]
                        current_list_inner.extend(c_n)
                    current_list.append(current_list_inner)
                current_list1 = np.array(current_list)

                #current_list1 = current_list.transpose(1,0,2).reshape(8, -1)
                m_current = np.mean(current_list1, axis=0)

                norm_mean = np.sqrt(np.dot(m_current.T, m_current))
                #grad_norms = np.std(current_list1, axis=0)
                #lists_mean.append(np.mean(np.abs(np.mean(current_list1, axis=0)), axis=0))
                covs_mat = np.zeros((m_current.shape[0], m_current.shape[0]))
                print ('Calculating covarince...')
                #for neuron in range(current_list1.shape[0])
                for batch in range(current_list1.shape[0]):
                    current_vec = current_list1[batch, :] - m_current
                    current_cov_mat = np.dot(current_vec[:,None], current_vec[None,:])
                    covs_mat+=current_cov_mat
                covs_mat = np.array(covs_mat)/ current_list1.shape[0]
                #mean_cov_mat = np.mean(covs_mat, axis=0)
                mean_cov_mat = covs_mat
                trac_cov = np.trace(mean_cov_mat)
                lists_mean.append(norm_mean)
                lists.append(trac_cov)
                #Second method if we have a lot of neurons
                """
                #lists.append(np.mean(grad_norms))
                #lists_mean.append(norm_mean)
                for neuron in range(len(grad[epoch_number][0][layer])/10):
                    current_list = np.array([grad[epoch_number][i][layer][neuron] for i in range(len(grad[epoch_number]))])
                    total_w.extend(current_list.T)
                    grad_norms1 = np.std(current_list, axis=0)
                    mean_la = np.abs(np.mean(np.array(current_list), axis=0))
                    #mean_la = LA.norm(current_list, axis=0)
                    c_var.append(np.mean(grad_norms1))
                    c_mean.append(np.mean(mean_la))
                #total_w is in size [num_of_total_weights, num of epochs]
                total_w = np.array(total_w)
                #c_var.append(np.sqrt(np.trace(np.cov(np.array(total_w).T)))/np.cov(np.array(total_w).T).shape[0])
                #print np.mean(c_mean).shape
                lists_mean.append(np.mean(c_mean))
                lists.append(np.mean(c_var))
                """
            l2_norm_total.append(l2_norm)
            t_list_mean.append(np.array(lists_mean))
            t_lists.append((np.array(lists)))
        y = np.mean(np.array(t_lists), axis=0)/np.mean(np.array(l2_norm_total), axis=0)
        axes_norms.plot(epochsInds, np.mean(np.array(l2_norm_total), axis=0),linewidth = 2)
        y_mean = np.mean(np.array(t_list_mean), axis=0)/np.mean(np.array(l2_norm_total), axis=0)
        c_p1, = axes_log.plot(epochsInds, y,markersize = 4, linewidth = 4,color = colors[layer_index], linestyle=':', markeredgewidth=0.2, dashes = [4,4])
        c_p0,= axes_log.plot(epochsInds, y_mean,  linewidth = 2,color = colors[layer_index])
        #For the legend
        p_0.append(c_p0)
        p_1.append(c_p1)

    #adejust the figure according the specipic labels, scaling and legends
    updateAxes(axes_log, '# Epochs', 'Normalized Mean and STD', [0, 7000], [0.000001, 10], '', 'log', 'log', [1, 10, 100, 1000, 7000],  [0.00001, 0.0001,0.001,0.01,0.1, 1, 10], p_0, p_1)
    updateAxesNorms(axes_norms, '# Epochs', '$L_2$')
    f_log.savefig('log_gradient1.svg', dpi=200, format= 'svg')
    f_norms.savefig('norms.jpg', dpi=200, format= 'jpg')

if __name__ == '__main__':
    TIME_STEMPS = 'time-stemp'
    ALL_LAYERS = 'all_layers'
    COMPRAED_PERCENT = 'compare_percent'
    ALL_SAMPLES = 'all_samples'
    action = TIME_STEMPS
    prex = 'jobsFiles/'
    sofix = '.pickle'
    prex2 = 'jobs/'
    #str_names = [[prex2+'fo_layersSizes=[[10, 7, 5, 4, 3]]_LastEpochsInds=9998_numRepeats=1_batch=3563_DataName=reg_1_numEphocs=10000_learningRate=0.0004_numEpochsInds=964_samples=1_num_of_disribuation_samples=1/']]

    if action == TIME_STEMPS:
        index = 1
        save_name = '3_time_series'
        name_s = 'g_layersSizes=[[10, 7, 5, 4, 3]]_LastEpochsInds=9998_numRepeats=40_batch=3563_DataName=var_u_numEphocs=10000_learningRate=0.0002_numEpochsInds=964_samples=1_num_of_disribuation_samples=1/'
        printTimeFigures(name_s, save_name, index)
        plotAnimationAll(name_s,save_name)
    else:
        if action ==ALL_LAYERS:
            mode =1
            save_name = ALL_LAYERS
            str_names = [[prex + 'ff3_5_198.pickle', prex+ 'ff3_4_198.pickle'],[prex + 'ff3_3_198.pickle',prex + 'ff3_2_198.pickle'],[prex + 'ff3_1_198.pickle',prex + 'ff4_1_10.pickle']]
            str_names[2][1] = prex2+'g_layersSizes=[[10, 7, 5, 4, 4, 3]]_LastEpochsInds=9998_numRepeats=20_batch=3563_DataName=var_u_numEphocs=10000_learningRate=0.0004_numEpochsInds=964_samples=1_num_of_disribuation_samples=1/'
        elif action == COMPRAED_PERCENT:
            save_name = COMPRAED_PERCENT
            mode =2
            str_names    = [[prex + 'ff4_1_10.pickle', prex + 'ff3_1_198.pickle']]
        elif action == ALL_SAMPLES:
            save_name = ALL_SAMPLES
            mode =3
            str_names = [[prex+'t_32_1.pickle']]
        plotFigures(str_names, mode,save_name)
        plotNorms(str_names)
        plotGrads(str_names)
        calcvVarNeurons(str_names)
        plotLayers(str_names, save_name)
    plt.show()



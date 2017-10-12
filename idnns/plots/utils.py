import matplotlib

matplotlib.use("TkAgg")
import scipy.io as sio

import matplotlib.pyplot as plt
import os
import numpy as np
import sys

if sys.version_info >= (3, 0):
	import _pickle as cPickle
else:
	import cPickle


def update_axes(axes, f, xlabel, ylabel, xlim, ylim, title='', xscale=None, yscale=None, x_ticks=None, y_ticks=None,
                p_0=None, p_1=None, p_3=None, p_4=None,
                p_5=None, title_size=22):
	"""adjust the axes to the ight scale/ticks and labels"""
	font_size = 30
	axis_font = 25
	legend_font = 16
	categories = 6 * ['']
	labels = ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$']
	# If we want grey line in the midle
	# axes.axvline(x=370, color='grey', linestyle=':', linewidth = 4)
	# The legents of the mean and the std
	"""
	if p_0:
		leg1 = f.legend([p_0[0],p_0[1],p_0[2],p_0[3],p_0[4], p_0[5]], categories, title=r'$\|Mean\left(\nabla{W_i}\right)\|$',bbox_to_anchor=(0.09, 0.95),  loc=2,fontsize = legend_font,markerfirst = False, handlelength = 5)
		leg1.get_title().set_fontsize('21')  # legend 'Title' fontsize
		axes.add_artist(leg1)

	if p_1:
		leg2 = f.legend([p_1[0],p_1[1],p_1[2],p_1[3],p_1[4], p_1[5]], categories, title=r'$Variance\left(\nabla{W_i}\right)$', loc=2,bbox_to_anchor=(0.25, 0.95), fontsize = legend_font ,markerfirst = False,handlelength = 5)
		leg2.get_title().set_fontsize('21')  # legend 'Title' fontsize
		axes.add_artist(leg2)
	if p_3:
		leg2 = f.legend([p_3[0], p_3[1], p_3[2], p_3[3], p_3[4], p_3[5]], categories,
						  title=r'$SNR\left(\nabla{W_i}\right)$', loc=3, fontsize=legend_font,
						  markerfirst=False, handlelength=5,bbox_to_anchor=(0.15, 0.1))
		leg2.get_title().set_fontsize('21')  # legend 'Title'

	if p_4:
		leg2 = f.legend([p_4[0], p_4[1], p_4[2], p_4[3], p_4[4], p_4[5]], categories,
						title=r'$\log\left(1+ SNR\left(\nabla{W_i}\right)\right)$', loc=3, fontsize=legend_font,
						markerfirst=False, handlelength=5, bbox_to_anchor=(0.15, 0.1))
		leg2.get_title().set_fontsize('21')  # legend 'Title'
	if p_5:
		pass
		#leg2 = axes.legend(handles=[r'$\frac{|d Error|}{STD\left(Error)\right)}$'], loc=3, fontsize=legend_font,
		#                bbox_to_anchor=(0.15, 0.1))
		#leg2.get_title().set_fontsize('21')
	"""
	# plt.gca().add_artist(leg2)
	axes.set_xscale(xscale)
	axes.set_yscale(yscale)
	axes.set_xlabel(xlabel, fontsize=font_size)
	axes.set_ylabel(ylabel, fontsize=font_size)
	axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	if y_ticks:
		axes.set_xticks(x_ticks)
		axes.set_yticks(y_ticks)
	axes.tick_params(axis='x', labelsize=axis_font)
	axes.tick_params(axis='y', labelsize=axis_font)

	axes.xaxis.major.formatter._useMathText = True
	axes.set_yticklabels(labels, fontsize=font_size)
	axes.set_title(title, fontsize=title_size)
	axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
	axes.set_xlim(xlim)
	if ylim:
		axes.set_ylim(ylim)


def update_axes_norms(axes, xlabel, ylabel):
	"""Adjust the axes of the norms figure with labels/ticks"""
	font_size = 30
	axis_font = 25
	legend_font = 16
	# the legends
	categories = [r'$\|W_1\|$', r'$\|W_2\|$', r'$\|W_3\|$', r'$\|W_4\|$', r'$\|W_5\|$', r'$\|W_6\|$']
	# Grey line in the middle
	axes.axvline(x=370, color='grey', linestyle=':', linewidth=4)
	axes.legend(categories, loc='best', fontsize=legend_font)
	axes.set_xlabel(xlabel, fontsize=font_size)
	axes.set_ylabel(ylabel, fontsize=font_size)
	axes.tick_params(axis='x', labelsize=axis_font)
	axes.tick_params(axis='y', labelsize=axis_font)


def update_axes_snr(axes, xlabel, ylabel):
	"""Adjust the axes of the norms figure with labels/ticks"""
	font_size = 30
	axis_font = 25
	legend_font = 16
	# the legends
	categories = [r'$W_1$', r'$W_2$', r'$W_3$', r'$W_4$', r'$W_5$', r'$W_6$']
	# Grey line in the middle
	axes.set_title('The SNR ($norm^2/variance$)')
	# axes.axvline(x=370, color='grey', linestyle=':', linewidth=4)
	axes.legend(categories, loc='best', fontsize=legend_font)
	axes.set_xlabel(xlabel, fontsize=font_size)
	axes.set_ylabel(ylabel, fontsize=font_size)
	axes.tick_params(axis='x', labelsize=axis_font)
	axes.tick_params(axis='y', labelsize=axis_font)


def adjust_axes(axes_log, axes_norms, p_0, p_1, f_log, f_norms, axes_snr=None, f_snr=None, p_3=None, axes_gaus=None,
                f_gau=None, p_4=None, directory_name=''):
	# adejust the figure according the specipic labels, scaling and legends
	# Change the log and log to linear if you want linear scaling
	# update_axes(reg_axes, '# Epochs', 'Normalized Mean and STD', [0, 10000], [0.000001, 10], '', 'log', 'log', [1, 10, 100, 1000, 10000], [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], p_0, p_1)
	title = 'The Mean and std of the gradients of each layer'
	update_axes(axes_log, f_log, '# Epochs', 'Mean and STD', [0, 7000], [0.001, 10], title, 'log', 'log',
	            [1, 10, 100, 1000, 7000], [0.001, 0.01, 0.1, 1, 10], p_0, p_1)
	update_axes_norms(axes_norms, '# Epochs', '$L_2$')
	if p_3:
		title = r'SNR of the gradients ($\frac{norm^2}{variance}$)'
		update_axes(axes_snr, f_snr, '# Epochs', 'SNR', [0, 7000], [0.0001, 10], title, 'log', 'log',
		            [1, 10, 100, 1000, 7000], [0.0001, 0.001, 0.01, 0.1, 1, 10], p_3=p_3)
	if p_4:
		title = r'Gaussian Channel bounds of the gradients ($\log\left(1+SNR\right)$)'

		update_axes(axes_gaus, f_gau, '# Epochs', 'log(SNR+1)', [0, 7000], [0.0001, 10], title, 'log', 'log',
		            [1, 10, 100, 1000, 7000], [0.0001, 0.001, 0.01, 0.1, 1, 10], p_4=p_4)
	# axes_log.plot(epochsInds[1:], np.abs(np.diff(np.squeeze(data_array['loss_train']))) / np.diff(epochsInds[:]), color='black', linewidth = 3)

	# axes_log.plot(epochsInds[0:], np.sum(np.array(sum_y), axis=0), color='c', linewidth = 3)
	# axes_log.plot(epochsInds[1:], diff_mean_loss, color='red', linewidth = 3)
	# f_log1, (axes_log1) = plt.subplots(1, 1, figsize=fig_size)

	# axes_log1.plot(epochsInds[1:], np.sum(np.array(sum_y), axis=0)[1:] / diff_mean_loss, color='c', linewidth=3)

	# axes_log.set_xscale('log')
	f_log.savefig(directory_name + 'log_gradient.svg', dpi=200, format='svg')
	f_norms.savefig(directory_name + 'norms.jpg', dpi=200, format='jpg')


def adjustAxes(axes, axis_font=20, title_str='', x_ticks=[], y_ticks=[], x_lim=None, y_lim=None,
               set_xlabel=True, set_ylabel=True, x_label='', y_label='', set_xlim=True, set_ylim=True, set_ticks=True,
               label_size=20, set_yscale=False,
               set_xscale=False, yscale=None, xscale=None, ytick_labels='', genreal_scaling=False):
	"""Organize the axes of the given figure"""
	if set_xscale:
		axes.set_xscale(xscale)
	if set_yscale:
		axes.set_yscale(yscale)
	if genreal_scaling:
		axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		axes.xaxis.major.formatter._useMathText = True
		axes.set_yticklabels(ytick_labels)
		axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
	if set_xlim:
		axes.set_xlim(x_lim)
	if set_ylim:
		axes.set_ylim(y_lim)
	axes.set_title(title_str, fontsize=axis_font + 2)
	axes.tick_params(axis='y', labelsize=axis_font)
	axes.tick_params(axis='x', labelsize=axis_font)
	if set_ticks:
		axes.set_xticks(x_ticks)
		axes.set_yticks(y_ticks)
	if set_xlabel:
		axes.set_xlabel(x_label, fontsize=label_size)
	if set_ylabel:
		axes.set_ylabel(y_label, fontsize=label_size)


def create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title):
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
	sm._A = []
	cbar_ax = f.add_axes(colorbar_axis)
	cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
	cbar.ax.tick_params(labelsize=bar_font)
	cbar.set_label(title, size=bar_font)
	cbar.ax.text(0.5, -0.01, epochsInds[0], transform=cbar.ax.transAxes,
	             va='top', ha='center', size=bar_font)
	cbar.ax.text(0.5, 1.0, str(epochsInds[-1]), transform=cbar.ax.transAxes,
	             va='bottom', ha='center', size=bar_font)


def get_data(name):
	"""Load data from the given name"""
	gen_data = {}
	# new version
	if os.path.isfile(name + 'data.pickle'):
		curent_f = open(name + 'data.pickle', 'rb')
		d2 = cPickle.load(curent_f)
	# Old version
	else:
		curent_f = open(name, 'rb')
		d1 = cPickle.load(curent_f)
		data1 = d1[0]
		data = np.array([data1[:, :, :, :, :, 0], data1[:, :, :, :, :, 1]])
		# Convert log e to log2
		normalization_factor = 1 / np.log2(2.718281)
		epochsInds = np.arange(0, data.shape[4])
		d2 = {}
		d2['epochsInds'] = epochsInds
		d2['information'] = data / normalization_factor
	return d2


def load_reverese_annealing_data(name, max_beta=300, min_beta=0.8, dt=0.1):
	"""Load mat file of the reverse annealing data with the give params"""
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


def get_data(name):
	"""Load data from the given name"""
	gen_data = {}
	# new version
	if os.path.isfile(name + 'data.pickle'):
		curent_f = open(name + 'data.pickle', 'rb')
		d2 = cPickle.load(curent_f)
	# Old version
	else:
		curent_f = open(name, 'rb')
		d1 = cPickle.load(curent_f)
		data1 = d1[0]
		data = np.array([data1[:, :, :, :, :, 0], data1[:, :, :, :, :, 1]])
		# Convert log e to log2
		normalization_factor = 1 / np.log2(2.718281)
		epochsInds = np.arange(0, data.shape[4])
		d2 = {}
		d2['epochsInds'] = epochsInds
		d2['information'] = data / normalization_factor
	return d2

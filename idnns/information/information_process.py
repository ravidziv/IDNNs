'''
Calculate the information in the network
Can be by the full distribution rule (for small netowrk) or bt diffrenet approximation method
'''
import multiprocessing
import warnings
import numpy as np
import tensorflow as tf
import idnns.information.information_utilities as inf_ut
from idnns.networks import model as mo
from idnns.information.mutual_info_estimation import calc_varitional_information
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
from idnns.information.mutual_information_calculation import *


def calc_information_for_layer(data, bins, unique_inverse_x, unique_inverse_y, pxs, pys1):
	bins = bins.astype(np.float32)
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	return local_IXT, local_ITY


def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y, calc_DKL=False):
	bins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]
	# bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
	# hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	if calc_DKL:
		pxy_given_T = np.array(
			[calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))]
		)
		p_XgT = np.vstack(pxy_given_T[:, 0])
		p_YgT = pxy_given_T[:, 1]
		p_YgT = np.vstack(p_YgT).T
		DKL_YgX_YgT = np.sum([inf_ut.KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
		H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	return local_IXT, local_ITY


def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label,
                                          b, b1, len_unique_a, pxs, p_YgX, pys1,
                                          percent_of_sampling=50):
	local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
	                                                 len_unique_a, p_YgX, unique_inverse_x,
	                                                 unique_inverse_y)
	number_of_indexs = int(data.shape[1] * (1. / 100 * percent_of_sampling))
	indexs_of_sampls = np.random.choice(data.shape[1], number_of_indexs, replace=False)
	if percent_of_sampling != 100:
		sampled_data = data[:, indexs_of_sampls]
		sampled_local_IXT, sampled_local_ITY = calc_information_sampling(
			sampled_data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x, unique_inverse_y)

	params = {}
	params['local_IXT'] = local_IXT
	params['local_ITY'] = local_ITY
	return params


def calc_by_sampling_neurons(ws_iter_index, num_of_samples, label, sigma, bins, pxs):
	iter_infomration = []
	for j in range(len(ws_iter_index)):
		data = ws_iter_index[j]
		new_data = np.zeros((num_of_samples * data.shape[0], data.shape[1]))
		labels = np.zeros((num_of_samples * label.shape[0], label.shape[1]))
		x = np.zeros((num_of_samples * data.shape[0], 2))
		for i in range(data.shape[0]):
			cov_matrix = np.eye(data[i, :].shape[0]) * sigma
			t_i = np.random.multivariate_normal(data[i, :], cov_matrix, num_of_samples)
			new_data[num_of_samples * i:(num_of_samples * (i + 1)), :] = t_i
			labels[num_of_samples * i:(num_of_samples * (i + 1)), :] = label[i, :]
			x[num_of_samples * i:(num_of_samples * (i + 1)), 0] = i
		b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
		unique_array, unique_indices, unique_inverse_x, unique_counts = \
			np.unique(b, return_index=True, return_inverse=True, return_counts=True)
		b_y = np.ascontiguousarray(labels).view(np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
		unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
			np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
		pys1 = unique_counts_y / float(np.sum(unique_counts_y))
		iter_infomration.append(
			calc_information_for_layer(data=new_data, bins=bins, unique_inverse_x=unique_inverse_x,
			                           unique_inverse_y=unique_inverse_y, pxs=pxs, pys1=pys1))
		params = np.array(iter_infomration)
		return params


def calc_information_for_epoch(iter_index, interval_information_display, ws_iter_index, bins, unique_inverse_x,
                               unique_inverse_y, label, b, b1,
                               len_unique_a, pys, pxs, py_x, pys1, model_path, input_size, layerSize,
                               calc_vartional_information=False, calc_information_by_sampling=False,
                               calc_full_and_vartional=False, calc_regular_information=True, num_of_samples=100,
                               sigma=0.5, ss=[], ks=[]):
	"""Calculate the information for all the layers for specific epoch"""
	np.random.seed(None)
	if calc_full_and_vartional:
		# Vartional information
		params_vartional = [
			calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
			                            input_size, layerSize, ss[i], pys, ks[i], search_sigma=False) for i in
			range(len(ws_iter_index))]
		# Full plug-in infomration
		params_original = np.array(
			[calc_information_for_layer_with_other(data=ws_iter_index[i], bins=bins, unique_inverse_x=unique_inverse_x,
			                                       unique_inverse_y=unique_inverse_y, label=label,
			                                       b=b, b1=b1, len_unique_a=len_unique_a, pxs=pxs,
			                                       p_YgX=py_x, pys1=pys1)
			 for i in range(len(ws_iter_index))])
		# Combine them
		params = []
		for i in range(len(ws_iter_index)):
			current_params = params_original[i]
			current_params_vartional = params_vartional[i]
			current_params['IXT_vartional'] = current_params_vartional['local_IXT']
			current_params['ITY_vartional'] = current_params_vartional['local_ITY']
			params.append(current_params)
	elif calc_vartional_information:
		params = [
			calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
			                            input_size, layerSize, ss[i], pys, ks[i], search_sigma=True) for i in
			range(len(ws_iter_index))]
	# Calc infomration of only subset of the neurons
	elif calc_information_by_sampling:
		parmas = calc_by_sampling_neurons(ws_iter_index=ws_iter_index, num_of_samples=num_of_samples, label=label,
		                                  sigma=sigma, bins=bins, pxs=pxs)

	elif calc_regular_information:
		params = np.array(
			[calc_information_for_layer_with_other(data=ws_iter_index[i], bins=bins, unique_inverse_x=unique_inverse_x,
			                                       unique_inverse_y=unique_inverse_y, label=label,
			                                       b=b, b1=b1, len_unique_a=len_unique_a, pxs=pxs,
			                                       p_YgX=py_x, pys1=pys1)
			 for i in range(len(ws_iter_index))])

	if np.mod(iter_index, interval_information_display) == 0:
		print('Calculated The information of epoch number - {0}'.format(iter_index))
	return params


def extract_probs(label, x):
	"""calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
	pys = np.sum(label, axis=0) / float(label.shape[0])
	b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
	unique_array, unique_indices, unique_inverse_x, unique_counts = \
		np.unique(b, return_index=True, return_inverse=True, return_counts=True)
	unique_a = x[unique_indices]
	b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
	pxs = unique_counts / float(np.sum(unique_counts))
	p_y_given_x = []
	for i in range(0, len(unique_array)):
		indexs = unique_inverse_x == i
		py_x_current = np.mean(label[indexs, :], axis=0)
		p_y_given_x.append(py_x_current)
	p_y_given_x = np.array(p_y_given_x).T
	b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
	unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
		np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
	pys1 = unique_counts_y / float(np.sum(unique_counts_y))
	return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs


def get_information(ws, x, label, num_of_bins, interval_information_display, model, layerSize,
                    calc_parallel=True, py_hats=0):
	"""Calculate the information for the network for all the epochs and all the layers"""
	print('Start calculating the information...')
	bins = np.linspace(-1, 1, num_of_bins)
	label = np.array(label).astype(np.float)
	pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)
	if calc_parallel:
		params = np.array(Parallel(n_jobs=NUM_CORES
		                           )(delayed(calc_information_for_epoch)
		                             (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
		                              label,
		                              b, b1, len(unique_a), pys,
		                              pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize)
		                             for i in range(len(ws))))
	else:
		params = np.array([calc_information_for_epoch
		                   (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
		                    label, b, b1, len(unique_a), pys,
		                    pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize)
		                   for i in range(len(ws))])
	return params

'Calculation of the full plug-in distribuation'

import numpy as np
import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


def calc_entropy_for_specipic_t(current_ts, px_i):
	"""Calc entropy for specipic t"""
	b2 = np.ascontiguousarray(current_ts).view(
		np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_current_ts = unique_counts / float(sum(unique_counts))
	p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
	H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
	return H2X


def calc_condtion_entropy(px, t_data, unique_inverse_x):
	# Condition entropy of t given x
	H2X_array = np.array(
		Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
		                           for i in range(px.shape[0])))
	H2X = np.sum(H2X_array)
	return H2X


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
	"""Calculate the MI based on binning of the data"""
	H2 = -np.sum(ps2 * np.log2(ps2))
	H2X = calc_condtion_entropy(px, data, unique_inverse_x)
	H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
	IY = H2 - H2Y
	IX = H2 - H2X
	return IX, IY


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
	"""Calculate the p(x|T) and p(y|T)"""
	indexs = unique_inverse == t_index
	p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
	unique_array_internal, unique_counts_internal = \
		np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
	indexes_x = np.where(np.in1d(b1, b[indexs]))
	p_x_ts = np.zeros(len_unique_a)
	p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
	return p_x_ts, p_y_ts

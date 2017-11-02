import numpy as np
from scipy.optimize import minimize
import sys
import tensorflow as tf
from idnns.networks import model as mo
import contextlib
import idnns.information.entropy_estimators as ee

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def optimiaze_func(s, diff_mat, d, N):
    diff_mat1 = (1. / (np.sqrt(2. * np.pi) * (s ** 2) ** (d / 2.))) * np.exp(-diff_mat / (2. * s ** 2))
    np.fill_diagonal(diff_mat1, 0)
    diff_mat2 = (1. / (N - 1)) * np.sum(diff_mat1, axis=0)
    diff_mat3 = np.sum(np.log2(diff_mat2), axis=0)
    return -diff_mat3


def calc_all_sigams(data, sigmas):
    batchs = 128
    num_of_bins = 8
    # bins = np.linspace(-1, 1, num_of_bins).astype(np.float32)
    # bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
    # data = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)

    batch_points = np.rint(np.arange(0, data.shape[0] + 1, batchs)).astype(dtype=np.int32)
    I_XT = []
    num_of_rand = min(800, data.shape[1])
    for sigma in sigmas:
        # print sigma
        I_XT_temp = 0
        for i in range(0, len(batch_points) - 1):
            new_data = data[batch_points[i]:batch_points[i + 1], :]
            rand_indexs = np.random.randint(0, new_data.shape[1], num_of_rand)
            new_data = new_data[:, :]
            N = new_data.shape[0]
            d = new_data.shape[1]
            diff_mat = np.linalg.norm(((new_data[:, np.newaxis, :] - new_data)), axis=2)
            # print diff_mat.shape, new_data.shape
            s0 = 0.2
            # DOTO -add leaveoneout validation
            res = minimize(optimiaze_func, s0, args=(diff_mat, d, N), method='nelder-mead',
                           options={'xtol': 1e-8, 'disp': False, 'maxiter': 6})
            eta = res.x
            diff_mat0 = - 0.5 * (diff_mat / (sigma ** 2 + eta ** 2))
            diff_mat1 = np.sum(np.exp(diff_mat0), axis=0)
            diff_mat2 = -(1.0 / N) * np.sum(np.log2((1.0 / N) * diff_mat1))
            I_XT_temp += diff_mat2 - d * np.log2((sigma ** 2) / (eta ** 2 + sigma ** 2))
            # print diff_mat2 - d*np.log2((sigma**2)/(eta**2+sigma**2))
        I_XT_temp /= len(batch_points)
        I_XT.append(I_XT_temp)
    sys.stdout.flush()
    return I_XT


def estimate_IY_by_network(data, labels, from_layer=0):
    if len(data.shape) > 2:
        input_size = data.shape[1:]
    else:
        input_size = data.shape[1]
    p_y_given_t_i = data
    acc_all = [0]
    if from_layer < 5:

        acc_all = []
        g1 = tf.Graph()  ## This is one graph
        with g1.as_default():
            # For each epoch and for each layer we calculate the best decoder - we train a 2 lyaer network
            cov_net = 4
            model = mo.Model(input_size, [400, 100, 50], labels.shape[1], 0.0001, '', cov_net=cov_net,
                             from_layer=from_layer)
            if from_layer < 5:
                optimizer = model.optimize
            init = tf.global_variables_initializer()
            num_of_ephocs = 50
            batch_size = 51
            batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
            if data.shape[0] not in batch_points:
                batch_points = np.append(batch_points, [data.shape[0]])
        with tf.Session(graph=g1) as sess:
            sess.run(init)
            if from_layer < 5:
                for j in range(0, num_of_ephocs):
                    for i in range(0, len(batch_points) - 1):
                        batch_xs = data[batch_points[i]:batch_points[i + 1], :]
                        batch_ys = labels[batch_points[i]:batch_points[i + 1], :]
                        feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                        if cov_net == 1:
                            feed_dict[model.drouput] = 0.5
                        optimizer.run(feed_dict)
            p_y_given_t_i = []
            batch_size = 256
            batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
            if data.shape[0] not in batch_points:
                batch_points = np.append(batch_points, [data.shape[0]])
            for i in range(0, len(batch_points) - 1):
                batch_xs = data[batch_points[i]:batch_points[i + 1], :]
                batch_ys = labels[batch_points[i]:batch_points[i + 1], :]
                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                if cov_net == 1:
                    feed_dict[model.drouput] = 1
                p_y_given_t_i_local, acc = sess.run([model.prediction, model.accuracy],
                                                    feed_dict=feed_dict)
                acc_all.append(acc)
                if i == 0:
                    p_y_given_t_i = np.array(p_y_given_t_i_local)
                else:
                    p_y_given_t_i = np.concatenate((p_y_given_t_i, np.array(p_y_given_t_i_local)), axis=0)
                    # print ("The accuracy of layer number - {}  - {}".format(from_layer, np.mean(acc_all)))
    max_indx = len(p_y_given_t_i)
    labels_cut = labels[:max_indx, :]
    true_label_index = np.argmax(labels_cut, 1)
    s = np.log2(p_y_given_t_i[np.arange(len(p_y_given_t_i)), true_label_index])
    I_TY = np.mean(s[np.isfinite(s)])
    PYs = np.sum(labels_cut, axis=0) / labels_cut.shape[0]
    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))
    I_TY = Hy + I_TY
    I_TY = I_TY if I_TY >= 0 else 0
    acc = np.mean(acc_all)
    sys.stdout.flush()
    return I_TY, acc


def calc_varitional_information(data, labels, model_path, layer_numer, num_of_layers, epoch_index, input_size,
                                layerSize, sigma, pys, ks,
                                search_sigma=False, estimate_y_by_network=False):
    """Calculate estimation of the information using vartional IB"""
    # Assumpations
    estimate_y_by_network = True
    # search_sigma = False
    data_x = data.reshape(data.shape[0], -1)

    if search_sigma:
        sigmas = np.linspace(0.2, 10, 20)
        sigmas = [0.2]

    else:
        sigmas = [sigma]
    if False:
        I_XT = calc_all_sigams(data_x, sigmas)
    else:
        I_XT = 0
    if estimate_y_by_network:

        I_TY, acc = estimate_IY_by_network(data, labels, from_layer=layer_numer)
    else:
        I_TY = 0
    with printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
        print('[{0}:{1}] - I(X;T) - {2}, I(X;Y) - {3}, accuracy - {4}'.format(epoch_index, layer_numer,
                                                                              np.array(I_XT).flatten(), I_TY, acc))
    sys.stdout.flush()

    # I_est = mutual_inform[ation((data, labels[:, 0][:, None]), PYs, k=ks)
    # I_est,I_XT = 0, 0
    params = {}
    # params['DKL_YgX_YgT'] = DKL_YgX_YgT
    # params['pts'] = p_ts
    # params['H_Xgt'] = H_Xgt
    params['local_IXT'] = I_XT
    params['local_ITY'] = I_TY
    return params

def estimate_Information(Xs, Ys, Ts):
	"""Estimation of the MI from missing data based on k-means clustring"""
	estimate_IXT = ee.mi(Xs, Ts)
	estimate_IYT = ee.mi(Ys, Ts)
	# estimate_IXT1 = ee.mi(Xs, Ts)
	# estimate_IYT1 = ee.mi(Ys, Ts)
	return estimate_IXT, estimate_IYT


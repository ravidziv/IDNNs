import multiprocessing
from idnns.information.mutual_info_estimation import mutual_information
import entropy_estimators as ee
import idnns.information.information_utilities as inf_ut
import numpy as np
import tensorflow as tf
from idnns import model as mo
import  warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
from scipy.optimize import minimize


def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts* np.log2(p_current_ts)))
    return H2X

def calc_condtion_entropy(px, t_data, unique_inverse_x):
    #Condition entropy of t given x
    H2X_array = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i , :], px[i])
                                                    for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x,unique_inverse_y,unique_array):
    """Calculate the MI based on binning of the data"""
    H2 = -np.sum(ps2* np.log2(ps2))
    H2X = calc_condtion_entropy(px, data ,unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data ,unique_inverse_y)
    IY=H2 - H2Y
    IX = H2 - H2X
    return IX, IY


def estimate_Information(Xs, Ys, Ts):
    """Estimation of the MI from missing data based on k-means clustring"""
    estimate_IXT = ee.mi(Xs, Ts)
    estimate_IYT = ee.mi(Ys, Ts)
    #estimate_IXT1 = ee.mi(Xs, Ts)
    #estimate_IYT1 = ee.mi(Ys, Ts)
    return estimate_IXT, estimate_IYT


def get_infomration_specipic(ws_local, x, label,estimted_label, num_of_bins,interval_information_display, indexs, model):
    """Calculate the information for the network"""
    estimted_label = [local_estimted_label[:,0][None,:] for local_estimted_label in estimted_label]
    #return getProb_spec(ws_local, x, label, estimted_label, num_of_bins,interval_information_display,model)

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


def calc_information_for_layer(data, bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1, len_unique_a, pys, py_hats, pxs, py_x, layer_index, epoch_index, pys1):
    bins = bins.astype(np.float32)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y, unique_array)

    return local_IXT, local_ITY

def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX , unique_inverse_x, unique_inverse_y):

    bins = bins.astype(np.float32)
    num_of_bins = bins.shape[0]
    from scipy import stats
    # bins = stats.mstats.mquantiles(np.squeeze(data.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
    # hist, bin_edges = np.histogram(np.squeeze(data.reshape(1, -1)), normed=True)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
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
    return local_IXT, local_ITY,DKL_YgX_YgT,p_ts,H_Xgt

def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label, estimted_label,
                                          b, b1, len_unique_a, pys, py_hats, pxs, p_YgX, layer_index, epoch_index, pys1, percent_of_sampling = 50):
    local_IXT, local_ITY,DKL_YgX_YgT,p_ts,H_Xgt = calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX , unique_inverse_x, unique_inverse_y)
    number_of_indexs = int(data.shape[1] * (1. / 100 * percent_of_sampling))
    indexs_of_sampls = np.random.choice(data.shape[1], number_of_indexs, replace=False)
    if percent_of_sampling !=100:
        sampled_data = data[:, indexs_of_sampls]
        sampled_local_IXT, sampled_local_ITY,sampled_DKL_YgX_YgT,sampled_p_ts,sampled_H_Xgt = calc_information_sampling(sampled_data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX , unique_inverse_x, unique_inverse_y)


    params={}
    params['DKL_YgX_YgT'] = DKL_YgX_YgT
    params['pts'] = p_ts
    params['H_Xgt'] = H_Xgt
    params['local_IXT'] = local_IXT
    params['local_ITY'] = local_ITY
    params['local_IXT_sampled'] = sampled_local_IXT
    params['local_ITY_sampled'] = sampled_local_ITY
    return params


def calc_information_for_epoch(iter_index, interval_information_display, ws_iter_index, bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                               len_unique_a, pys, py_hats, pxs, py_x, pys1, model_path, input_size, layerSize,
                               calc_vartional_information=False, calc_information_by_sampling=False,calc_combined = False,calc_regular_information = True):
    """Calculate the information for all the layers for specific epoch"""
    np.random.seed(None)
    if calc_combined:
        ss = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        ks = [4, 15, 50, 50, 50, 100]
        params_vartional = [
            calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
                                        input_size, layerSize, ss[i], pys, ks[i],search_sigma=False) for i in range(len(ws_iter_index))]

        params_original = np.array(
            [calc_information_for_layer_with_other(ws_iter_index[i], bins, unique_inverse_x, unique_inverse_y, label,
                                                   estimted_label, b, b1,
                                                   len_unique_a, pys, py_hats, pxs, py_x, i, iter_index, pys1)
             for i in range(len(ws_iter_index))])
        params = []
        for i in range(len(ws_iter_index)):
            current_params = params_original[i]
            current_params_vartional = params_vartional[i]
            current_params['IXT_vartional'] = current_params_vartional['local_IXT']
            current_params['ITY_vartional'] = current_params_vartional['local_ITY']
            params.append(current_params)

    elif calc_vartional_information:
        #TODO - optimze this values
        ss = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        ks = [4, 15,50,50,50,100]
        params = [
            calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
                                     input_size, layerSize, ss[i], pys,ks[i],search_sigma=True) for i in range(len(ws_iter_index))]
    elif calc_information_by_sampling:
        # go over all layers
        iter_infomration = []
        num_of_samples = 100
        for i in range(len(ws_iter_index)):
            data = ws_iter_index[i]
            new_data = np.zeros((num_of_samples * data.shape[0], data.shape[1]))
            labels = np.zeros((num_of_samples * label.shape[0], label.shape[1]))
            x = np.zeros((num_of_samples * data.shape[0], 2))
            sigma = 0.5
            for i in range(data.shape[0]):
                print i
                cov_matrix = np.eye(data[i, :].shape[0]) * sigma
                t_i = np.random.multivariate_normal(data[i, :], cov_matrix, num_of_samples)
                new_data[num_of_samples * i:(num_of_samples * (i + 1)) , :] = t_i
                labels[num_of_samples * i:(num_of_samples * (i + 1)) , :] = label[i, :]
                x[num_of_samples * i:(num_of_samples * (i + 1)) , 0] = i

            b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
            unique_array, unique_indices, unique_inverse_x, unique_counts = \
                np.unique(b, return_index=True, return_inverse=True, return_counts=True)
            b_y = np.ascontiguousarray(labels).view(np.dtype((np.void, labels.dtype.itemsize * labels.shape[1])))
            unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
                np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
            pys1 = unique_counts_y / float(np.sum(unique_counts_y))

            iter_infomration.append(calc_information_for_layer(new_data, bins, unique_inverse_x, unique_inverse_y, labels,
                                           estimted_label, b, b1,
                                           len_unique_a, pys, py_hats, pxs, py_x, i, iter_index, pys1))
            params = np.array(iter_infomration)
    elif calc_regular_information:
        params = np.array(
            [calc_information_for_layer_with_other(ws_iter_index[i], bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                                        len_unique_a, pys, py_hats, pxs, py_x, i, iter_index, pys1)
             for i in range(len(ws_iter_index))])

    if np.mod(iter_index, interval_information_display) == 0:
        print ('Calculated The information of epoch number - {0}'.format(iter_index))
    return params


def get_information(ws, x, label, estimted_label, num_of_bins, interval_information_display, indexs, model, layerSize, calc_parallel = True, py_hats = 0):
    """Calculate the information for the network for all the epochs and all the layers"""
    print ('Start calculating the information...')
    bins = np.linspace(-1, 1, num_of_bins)
    label = np.array(label).astype(np.float)
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    #prob_y_hat = [np.sum(estimted_label_local) / float(estimted_label_local.shape[1]) for estimted_label_local in estimted_label]
    #py_hats = [[local_p_y_hat, 1 - local_p_y_hat] for local_p_y_hat in prob_y_hat]
    pxs = unique_counts / float(np.sum(unique_counts))
    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs,:], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    if calc_parallel:
        params = np.array(Parallel(n_jobs=NUM_CORES
                                        )(delayed(calc_information_for_epoch)
                                                          (i,interval_information_display, ws[i], bins,unique_inverse_x,unique_inverse_y,label,estimted_label,
                                                           b, b1,len(unique_a), pys, py_hats,
                                                           pxs,p_y_given_x,pys1, model.save_file,  x.shape[1], layerSize)
                                               for i in range( len(ws))))
    else:
        params = np.array([calc_information_for_epoch
                                          (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
                                           label, estimted_label[i], b, b1, len(unique_a), pys, py_hats,
                                           pxs, p_y_given_x, pys1,model.save_file,  x.shape[1], layerSize)
                                for i in range(len(ws))])
    return params


def optimiaze_func(s,diff_mat,d, N):
    diff_mat1 = (1./(np.sqrt(2.*np.pi)*(s**2)**(d/2.)))*np.exp(-diff_mat /(2.*s**2))
    np.fill_diagonal(diff_mat1, 0)
    diff_mat2 = (1./(N-1))*np.sum(diff_mat1,axis=0)
    diff_mat3 = np.sum(np.log2(diff_mat2), axis=0)
    return -diff_mat3


def calc_all_sigams(data, sigmas):
    batchs = 512
    batch_points = np.rint(np.arange(0, data.shape[0] + 1, batchs)).astype(dtype=np.int32)
    I_XT = []
    for sigma in sigmas:
        #print sigma
        I_XT_temp = 0
        for i in xrange(0, len(batch_points) - 1):
            new_data = data[batch_points[i]:batch_points[i + 1], :]
            N = new_data.shape[0]
            d = new_data.shape[1]
            diff_mat = np.linalg.norm(((new_data[:, np.newaxis, :] - new_data)), axis=2)
            s0 = 0.2
            # DOTO -add leaveoneout validation

            res = minimize(optimiaze_func, s0, args=(diff_mat, d, N), method='nelder-mead',
                           options={'xtol': 1e-8, 'disp': False, 'maxiter': 10})
            eta = res.x
            diff_mat0 = - 0.5 * (diff_mat / (sigma ** 2 + eta ** 2))
            diff_mat1 = np.sum(np.exp(diff_mat0), axis=0)
            diff_mat2 = -(1.0 / N) * np.sum(np.log2((1.0 / N) * diff_mat1))
            I_XT_temp += diff_mat2 - d * np.log2((sigma ** 2) / (eta ** 2 + sigma ** 2))
            # print diff_mat2 - d*np.log2((sigma**2)/(eta**2+sigma**2))
        I_XT_temp /= len(batch_points)
        # print I_XT_temp
        I_XT.append(I_XT_temp)
    return I_XT


def estimate_IY_by_network(data, labels):
    tf.reset_default_graph()
    input_size = data.shape[1]
    # For each epoch and for each layer we calculate the best decoder - we train a 2 lyaer network
    model = mo.Model(input_size, [400, 100, 50], labels.shape[1], 0.001, '', cov_net=0)
    optimizer = model.optimize
    init = tf.global_variables_initializer()
    num_of_ephocs = 60

    batch_size = 1024
    batch_points = np.rint(np.arange(0, data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
    PYs = np.sum(labels, axis=0) / labels.shape[0]
    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))
    print ('rrrrr')
    with tf.Session() as sess:
        sess.run(init)
        # Go over the epochs
        for j in range(0, num_of_ephocs):
            for i in xrange(0, len(batch_points) - 1):
                batch_xs = data[batch_points[i]:batch_points[i + 1]]
                batch_ys = labels[batch_points[i]:batch_points[i + 1]]
                feed_dict = {model.x: batch_xs, model.labels: batch_ys}
                optimizer.run(feed_dict)
            #if np.mod(j, 1000) == 1:
        feed_dict = {model.x: data, model.labels: labels}
        print (j, sess.run( model.accuracy, feed_dict=feed_dict))
        # for i in range(data.shape[0]):
        feed_dict = {model.x: np.matrix(data)}
        p_y_given_t_i = sess.run(model.prediction, feed_dict=feed_dict)
        true_label_index = np.argmax(labels, 1)
        s = np.log2(p_y_given_t_i[np.arange(len(p_y_given_t_i)), true_label_index])
        I_TY = np.mean(s[np.isfinite(s)])
        sess.close()
    print ('jjj')

    I_TY = Hy + I_TY
    I_TY = I_TY if I_TY >= 0 else 0
    return I_TY
def calc_varitional_information(data, labels, model_path, layer_numer, num_of_layers, epoch_index,input_size, layerSize, sigma,pys,ks,
                                search_sigma = False,estimate_y_by_network=False):
    """Calculate estimation of the information using vartional IB"""
    #Assumpations -
    print 'kkkk'
    if search_sigma:
        sigmas = np.linspace(0.01, 0.15,2)
        #sigmas  = [sigmas[2]]
    else:
        sigmas = [sigma]
    I_XT = calc_all_sigams(data, sigmas)
    print 'ssss'
    if estimate_y_by_network:
        I_TY = estimate_IY_by_network(data, labels)
    else:
        I_TY = 0
    print ('ttt')
    #I_est = mutual_information((data, labels[:, 0][:, None]), PYs, k=ks)
    #I_est,I_XT = 0, 0
    params = {}
    #params['DKL_YgX_YgT'] = DKL_YgX_YgT
    #params['pts'] = p_ts
    #params['H_Xgt'] = H_Xgt
    params['local_IXT'] = I_XT
    params['local_ITY'] = I_TY
    return params

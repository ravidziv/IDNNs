import multiprocessing
from idnns.information.mutual_info_estimation import mutual_information
import entropy_estimators as ee
import numpy as np
import tensorflow as tf
from idnns import model as mo
import  warnings
warnings.filterwarnings("ignore")
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
    #label = label[:,0][None,:]
    estimted_label = [local_estimted_label[:,0][None,:] for local_estimted_label in estimted_label]
    #return getProb_spec(ws_local, x, label, estimted_label, num_of_bins,interval_information_display,model)



def processInputIter(data, bins, unique_inverse_x, unique_inverse_y, label,estimted_label, b, b1, len_unique_a, pys,py_hats, pxs, py_x,layer_index,epoch_index,pys1):
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


def calc_information_for_epoch(iter_index, interval_information_display, ws_iter_index, bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                               len_unique_a, pys, py_hats, pxs, py_x, pys1, model_path, input_size, layerSize, calc_vartional_information=False):
    """Calculate the information for all the layers for specific epoch"""
    np.random.seed(None)
    if calc_vartional_information:
        ss = [0.2, 0.2, 0.3, 0.4, 0.4, 0.5]
        ks = [4, 15,50,50,50,100]
        iter_infomration = [
            calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
                                     input_size, layerSize, ss[i], pys,ks[i]) for i in range(len(ws_iter_index))]
    else:
        # go over all layers
        iter_infomration = np.array(
            [processInputIter(ws_iter_index[i], bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                              len_unique_a, pys, py_hats, pxs, py_x, i, iter_index, pys1)
             for i in range(len(ws_iter_index))])
    if np.mod(iter_index, interval_information_display) == 0:
        print ('Calculated The information of epoch number - {0}'.format(iter_index))
    return iter_infomration


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
        infomration = np.array(Parallel(n_jobs=NUM_CORES
                                        )(delayed(calc_information_for_epoch)
                                                          (i,interval_information_display, ws[i], bins,unique_inverse_x,unique_inverse_y,label,estimted_label,
                                                           b, b1,len(unique_a), pys, py_hats,
                                                           pxs,p_y_given_x,pys1, model.save_file,  x.shape[1], layerSize)
                                               for i in range( len(ws))))
    else:
        infomration = np.array([calc_information_for_epoch
                                          (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
                                           label, estimted_label[i], b, b1, len(unique_a), pys, py_hats,
                                           pxs, p_y_given_x, pys1,model.save_file,  x.shape[1], layerSize)
                                for i in range(len(ws))])
    return infomration


def optimiaze_func(s,diff_mat,d, N):
    diff_mat1 = (1./(np.sqrt(2.*np.pi)*(s**2)**(d/2.)))*np.exp(-diff_mat /(2.*s**2))
    np.fill_diagonal(diff_mat1, 0)
    diff_mat2 = (1./(N-1))*np.sum(diff_mat1,axis=0)
    diff_mat3 = np.sum(np.log2(diff_mat2), axis=0)
    return -diff_mat3

def calc_varitional_information(data, labels, model_path, layer_numer, num_of_layers, epoch_index,input_size, layerSize, sigma,pys,ks):
    #Assumpations -
    #calc I(X;T)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    from scipy.optimize import minimize
    #print ('f', layer_numer)
    N = data.shape[0]
    #sigma = 0.5
    #eta = 0.5
    d = data.shape[1]
    diff_mat = np.linalg.norm(((data[:,np.newaxis,:]-data)), axis=2)
    s0 = 0.2
    #DOTO -add leaveoneout validation
    res = minimize(optimiaze_func, s0, args=(diff_mat, d, N), method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': False, 'maxiter':5})
    eta = res.x
    diff_mat0 = - 0.5 *(diff_mat / (sigma**2 + eta**2))
    diff_mat1 = np.sum(np.exp(diff_mat0), axis=0)
    diff_mat2 = -(1.0/N)*np.sum(np.log2((1.0/N)*diff_mat1))
    I_XT =diff_mat2 - d*np.log2((sigma**2)/(eta**2+sigma**2))
    I_TY = 0
    tf.reset_default_graph()
    model = mo.Model(input_size, layerSize, labels.shape[1])
    saver = tf.train.Saver()
    #print ('s', layer_numer)
    """"
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        #print '{0}-{1}.meta'.format(model_path,epoch_index)
        file_name = '{0}-{1}'.format(model_path,epoch_index)
        #saver = tf.train.import_meta_graph('{0}-{1}.meta'.format(model_path,epoch_index))
        #saver.restore(sess,'./{0}-{1}.meta'.format(model_path,epoch_index)')
        #with tf.Session() as sess:
        saver.restore(sess, file_name)

        #sess.run(tf.global_variables_initializer())
        #session = tf.get_default_session()
        #nputs = model.hidden_layers[layer_numer]
        #feed_dict = {self.x: X}
        #layer_values = session.run(inputs, feed_dict=feed_dict)


        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        #graph = tf.get_default_graph()
        #name_scope = 'hidden' + str(layer_numer)+ '_2:0'
        #name_scope = 'softmax_linear'
        #layer = graph.get_tensor_by_name(name_scope)
        # Now, access the op that you want to run.
        #preidcation = graph.get_tensor_by_name('hidden' +str(num_of_layers) +'_2:0')
        num_of_samples = 10
        new_data = np.zeros((num_of_samples*data.shape[0],data.shape[1] ))
        labels  = np.zeros((num_of_samples*labels.shape[0],labels.shape[1]))
        for i in range(data.shape[0]):
            #print (i, layer_numer)
            cov_matrix = np.eye(data[i,:].shape[0])*sigma
            t_i = np.random.multivariate_normal(data[i,:], cov_matrix)
            new_data[num_of_samples*i:(num_of_samples*(i+1))-1, :] = t_i
            labels[num_of_samples*i:(num_of_samples*(i+1))-1, :] = labels[i,:]
            '''
            #t_i = data[i,:]
            #feed_dict = {layer: 13.0}
            feed_dict = {model.hidden_layers[layer_numer]: np.matrix(t_i)}
            p_y_given_t_i = sess.run(model.prediction,  feed_dict=feed_dict)

            #p_y_given_t_i = sess.run(preidcation, feed_dict)
            #p_y_given_t_i = model.calc_values(t_i, layer_numer)
            true_label_index = np.argmax(labels[i,:])
            val = np.log2(p_y_given_t_i[0][true_label_index])
            if not np.isnan(val):
                I_TY += val
            '''
    """
    #print ('t', layer_numer)
    PYs = np.asarray(pys).T
    #I_est = mutual_information((data, labels[:, 0][:, None]), PYs, k=ks)
    I_est,I_XT = 0, 0
    #(I_TY / data.shape[0])
    return I_XT, I_est
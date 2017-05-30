import numpy as np
from idnns import entropy_estimators as ee
import warnings
#warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import multiprocessing
import tensorflow as tf
#from sympy.utilities.iterables import multiset_permutations
from idnns import model as mo

NUM_CORES = multiprocessing.cpu_count()


def calc_information(probTgivenXs, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))

    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))

    IYT = Hy - Hyt

    ITX = Ht - Htx
    return ITX, IYT

def calc_infomration_all_neruons(data, bins, label, b, b1, len_unique_a, pys, pxs, py_x):
    """Calculate the information for all the neuron in the layer"""

    binning_data = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    uni_array =np.array([np.unique(neuron_dist, return_index=False, return_inverse=True, return_counts=True)
        for neuron_dist in binning_data.T])
    unique_array, unique_inverse_t, unique_counts  = uni_array[:,0],uni_array[:,1] ,uni_array[:,2]
    p_ts =np.array([ unique_counts_per_neuron / float(sum(unique_counts_per_neuron))
        for unique_counts_per_neuron in unique_counts])
    I_XT, I_TY = [],[]
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys).T
    #go over all the neurons and add it to the infomration
    for i_n in range(len(p_ts)):
        current_unique_array = unique_array[i_n]
        current_unique_inverse_t = unique_inverse_t[i_n]
        pxys = np.array(
            [calc_probs(i, current_unique_inverse_t, label, label, b, b1, len_unique_a) for i in range(0, len(current_unique_array))]
        )
        p_x_given_t = pxys[:, 0]
        p_y_given_t = pxys[:, 1]
        p_x_given_t, p_y_given_t, p_y_given_x = np.vstack(p_x_given_t).T, np.vstack(p_y_given_t).T, np.vstack(
            py_x).T
        PTs = np.asarray(p_ts[i_n], dtype=np.float64).T
        PTgivenXs_not_divide =np.multiply(p_x_given_t,np.tile(PTs,(p_x_given_t.shape[0],1))).T
        PTgivenXs = np.multiply(PTgivenXs_not_divide, np.tile((1./(PXs)), (p_x_given_t.shape[1], 1)))

        local_IXT, local_ITY = process_and_calc_information_inner(PTgivenXs,p_y_given_t,PTs , PYs, PXs, p_y_given_x)
        I_XT.append(local_IXT)
        I_TY.append(local_ITY)
    return np.array(I_XT), np.array(I_TY)

def process_and_calc_information_inner_new(PTgivenXs, PYgivenTs, PYs, PXs, PTs):
    """Calculate I(X;T) and I(Y;T) """
    #PTs = np.asarray(PTs, dtype=np.float64).T
    #PTgivenXs_not_divide =np.multiply(PXgivenTs,np.tile(PTs,(PXgivenTs.shape[0],1))).T
    #PTgivenXs = np.multiply(PTgivenXs_not_divide, np.tile((1./(PXs)), (PXgivenTs.shape[1], 1)))
    ITX, IYT  = calc_information_new(PTgivenXs, PYgivenTs, PXs, PYs, PTs)
    return ITX, IYT

def calc_condtion_entropy(px, data ,unique_inverse_x):
    #H2X = 0
    H2X_array = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_one_iteration)(data[unique_inverse_x == i , :], px[i])
                                                    for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X


def calc_condtion_entropy_y(px, data ,unique_inverse_x):
    #H2X = 0
    H2X_array = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_one_iteration)(data[unique_inverse_x == 1-i , :], px[i])
                                                    for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X
def calc_one_iteration(current_ts,px_i ):
    #print(current_ts.shape)
    #current_ts = data[unique_inverse_x == i , :]
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    #p_current_ts = unique_counts/ float(np.sum(unique_counts)
    p_current_ts = unique_counts / float(sum(unique_counts))
    #p_current_ts  = np.asarray(p_current_ts).dtype
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts* np.log2(p_current_ts)))
    return H2X


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x,unique_inverse_y,unique_array):
    H2 = -np.sum(ps2* np.log2(ps2))
    H2X = calc_condtion_entropy(px, data ,unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data ,unique_inverse_y)
    """
    p_t_given_y = np.zeros((unique_array.shape[0], 2)).astype(np.float128)
    for i in range(2):
        current_ts =   data[unique_inverse_y == i, :]
        b2 = np.ascontiguousarray(current_ts).view(
            np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
        unique_index, unique_array, unique_inverse_t, unique_counts = \
            np.unique(b2, return_index=True, return_inverse=True, return_counts=True)
        p_ts_y_c = unique_counts.astype(np.float16) / np.sum(unique_counts.astype(np.float16))
        p_ts_y_c = p_ts_y_c.astype(np.float128)/ np.sum(p_ts_y_c.astype(np.float128))
        p_t_given_y[unique_index,1-i] = p_ts_y_c
    #HY = -np.sum(py* np.log2(py))
    #p_y_given_t = (p_t_given_y*py)/ps2
    H2Y_temp = - np.nansum(np.dot((p_t_given_y*np.log2(p_t_given_y+np.spacing(1))), py.T))
    """
    IY=H2 - H2Y
    IX = H2 - H2X
    return IX, IY

def calc_information_new(probTgivenXs, PYgivenTs, PXs, PYs,PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    #print ('fffff')
    #PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    index = PYgivenTs.shape[1] / 3
    probTgivenXsArray = np.split(probTgivenXs, [index, index * 2])
    Htx = 0
    for i in range(len(probTgivenXsArray)):
        #print ('j  -', i)
        sum_x_temp = - np.nansum((np.dot(probTgivenXsArray[-1]*np.log2(probTgivenXsArray[-1]), PXs)))
        Htx+= sum_x_temp
        del probTgivenXsArray[-1]
    #Htx = - np.nansum((np.dot(probTgivenXs*np.log2(probTgivenXs), PXs)))
    #print ('ssss')
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs + np.spacing(1)))

    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT

def process_and_calc_information_inner(PTgivenXs, PYgivenTs, PTs, PYs, PXs, PYgivenXs):
    """Calculate I(X;T) and I(Y;T) """

    ITX, IYT  = calc_information(PTgivenXs, PYgivenTs, PXs, PYs)
    return ITX, IYT


def calc_probs(t_index, unique_inverse, label,estimate_labels,  b, b1, len_unique_a):
    """Calculate the p(x|T) and p(y|T)"""
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
    p_y_hat_given_ts = np.sum(estimate_labels[indexs], axis=0) / estimate_labels[indexs].shape[0]

    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))

    return p_x_ts, p_y_ts,p_y_hat_given_ts


def estimate_Information(Xs, Ys, Ts):
    """Estimation of the MI from missing data based on k-means clustring"""
    estimate_IXT = ee.mi(Xs, Ts)
    estimate_IYT = ee.mi(Ys, Ts)
    #estimate_IXT1 = ee.mi(Xs, Ts)
    #estimate_IYT1 = ee.mi(Ys, Ts)
    return estimate_IXT, estimate_IYT


def get_infomration(ws, x, label,estimted_label, num_of_bins,interval_information_display, indexs, model):
    """Calculate the information for the network"""
    #label = label[:,0][None,:]
    #estimted_label = [local_estimted_label[:,0][None,:] for local_estimted_label in estimted_label]
    return getProb(ws, x, label, estimted_label, num_of_bins,interval_information_display,model)


def get_infomration_specipic(ws_local, x, label,estimted_label, num_of_bins,interval_information_display, indexs, model):
    """Calculate the information for the network"""
    #label = label[:,0][None,:]
    estimted_label = [local_estimted_label[:,0][None,:] for local_estimted_label in estimted_label]
    #return getProb_spec(ws_local, x, label, estimted_label, num_of_bins,interval_information_display,model)



def calcXI(probTgivenXs,PYgivenTs, PXs, PYs):
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT


def processInputProb(t_index, unique_inverse, label,estimted_label,  b, b1, len_unique_a):
    indexs = unique_inverse == t_index
    p_y_ts = np.mean(label[indexs, :], axis=0)
    #p_y_ts = np.array([p_y, 1 - p_y])

    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
    return p_x_ts, p_y_ts


def process_input_layer(t_index, unique_inverse, labels):
    #print t_index
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(labels[indexs,0]) / len(labels[indexs,0])
    #p_y_ts = np.array([p_y, 1 - p_y])
    return  p_y_ts



def processInputIter(data, bins, unique_inverse_x, unique_inverse_y, label,estimted_label, b, b1, len_unique_a, pys,py_hats, pxs, py_x,layer_index,epoch_index,pys1):
    bins = bins.astype(np.float32)
    from mutual_info import mutual_information

    #val, bins = np.histogram(data, density=True, bins=len(bins))
    from scipy import stats
    li = np.linspace(0,1, len(bins) )
    #bins = stats.mstats.mquantiles(data, li)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    #digitized1 = digitized / np.sum(digitized, axis=1)[:, None]
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))


    """"
    import timeit
    start = timeit.default_timer()

    pxys = np.array(Parallel(n_jobs=NUM_CORES)(delayed(processInputProb)
                                               (i, unique_inverse_t, label, estimted_label, b, b1, len_unique_a)
                                                for i in range(0, len(unique_array))))
    #p_t_y_hat = []

    #layer_values = []
    #for i in range(50):
    #    arr =np.random.permutation(digitized[:, :].T).T

    #    layer_values.append(arr)
    #layer_values = np.vstack(np.array(layer_values))

    #layer_values = 0
    #data_v = np.random.uniform(low=-1, high=1, size=(1000000,12))
    #labels_layers = model.inference(data_v)
    #labels_layers, layer_values = model.get_layer_with_inference(layer_values, layer_index, epoch_index)
    #bins_labels = np.linspace(0, 1, 20)

    #bins_values = np.linspace(-1, 1, 30)
    #labels_layers = bins_labels[np.digitize(np.squeeze(labels_layers.reshape(1, -1)), bins_labels) - 1].reshape(len(labels_layers), -1)
    #layer_values = bins_values[np.digitize(np.squeeze(layer_values.reshape(1, -1)), bins_values) - 1].reshape(len(layer_values), -1)

    #b2_layer = np.ascontiguousarray(layer_values).view(
    #    np.dtype((np.void, layer_values.dtype.itemsize * layer_values.shape[1])))
    #unique_array_layer, unique_inverse_layer, unique_counts_layer = \
    #   np.unique(b2_layer, return_index=False, return_inverse=True, return_counts=True)
    #p_ts_layer = unique_counts_layer / float(sum(unique_counts_layer))
    #labels_layers = model.inference(layer_values)
    #py_given_t__layer = np.array(Parallel(n_jobs=NUM_CORES)(delayed(process_input_layer)
    #    (i, unique_inverse_layer, labels_layers) for i in
    #     range(0, len(unique_array_layer))))
    #py_given_t__layer = np.array([py_given_t__layer, 1-py_given_t__layer]).T
    #p_y_hat_new = np.sum(labels_layers, axis=0)
    #p_y_hat_new = p_y_hat_new / np.sum(p_y_hat_new)


    PXgivenTs, PYgivenTs = np.vstack(pxys[:, 0]).T, np.vstack(pxys[:, 1]).T
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys).T
    PTs = np.asarray(p_ts, dtype=np.float64).T
    PTgivenXs_not_divide = np.multiply(PXgivenTs, np.tile(PTs, (PXgivenTs.shape[0], 1))).T
    PTgivenXs = np.multiply(PTgivenXs_not_divide, np.tile((1. / (PXs)), (PXgivenTs.shape[1], 1)))

    local_IXT1, local_ITY1= process_and_calc_information_inner\
        (PTgivenXs,PYgivenTs, PTs, PYs, PXs, py_x)
    stop = timeit.default_timer()
    #print stop - start
    start = timeit.default_timer()
"   """
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y, unique_array)
    """"
    p_t_given_x = np.zeros((p_ts.shape[0], PXs.shape[0]))
    # print ('before1')

    tg = (unique_inverse_t, np.arange(PXs.shape[0]))
    # print ('before1_2')

    p_t_given_x[tg] = 1

    index = p_ts.shape[0] / 3
    p_t_given_x0 = np.array_split(p_t_given_x,6)
    p_t0 = np.array_split(p_ts,6)
    # del p_t_given_x
    # p_t_given_x0[1]= p_t_given_x0[1]* PXs[None,:]
    # del p_t_given_x0
    p_y_given_t = []
    # tprint ('5555', len(p_t_given_x0))
    for i in range(len(p_t_given_x0)):
        # print (i)
        #p_t_given_x0[-1] = (p_t_given_x0[-1] * PXs[None, :]) / (p_t0[i][None,:])
        p_y_given_t.append(np.dot(py_x, ((p_t_given_x0[i] * PXs[None, :] )/ (p_t0[i][:,None])).T).T)
        #del p_t_given_x0[-1]
    p_y_given_t = np.vstack(p_y_given_t).T

    # PXgivenTs, PYgivenTs = np.vstack(pxys[:, 0]).T, np.vstack(pxys[:, 1]).T
    # p_t_given_x = np.zeros((p_ts.shape[0], pxs.shape[0]))
    # print ('after')
    local_IXT, local_ITY = process_and_calc_information_inner_new \
        (p_t_given_x, p_y_given_t, PYs, PXs, p_ts)
    #stop = timeit.default_timer()
    #print stop - start
    #print (local_IXT1, local_IXT, local_ITY1, local_ITY)
    """

    #if local_ITY<0 and local_ITY >-0.001:
    #    local_ITY = 0.0001
    return local_IXT, local_ITY


def processInput(iter_index, interval_information_display, ws_iter_index, bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                 len_unique_a, pys,py_hats, pxs, py_x,pys1, model_path,input_size, layerSize):
    #print (iter_index)
    np.random.seed(None)
    #if True or np.mod(iter_index, int(interval_information_display)) == 0:
    #    print ('1Calculated The information of epoch number - {0}'.format(iter_index))
    #print ('1 before')
    #print ('start - Calculated The information of epoch number - {0}'.format(iter_index))
    """Calculate the information for all the layers for specific epoch"""
        # go over all layers
    """"
    ss =[0.1, 0.2, 0.4, 0.6, 0.8, 1]
    iter_infomration= [calc_varitional_information(ws_iter_index[i], label, model_path, i,len(ws_iter_index)-1 , iter_index,input_size, layerSize, ss[i]) for i in range(len(ws_iter_index))]
    #ss.reverse()
    #print (iter_infomration   )

    #iter_infomration= [calc_varitional_information(ws_iter_index[i], label, model_path, i,len(ws_iter_index)-1 , iter_index,input_size, layerSize, ss[i]) for i in range(len(ws_iter_index))]
    print (iter_infomration   )
    ss =[0.01, 0.1, 0.2, 0.4, 0.5, 0.6]
    iter_infomration= [calc_varitional_information(ws_iter_index[i], label, model_path, i,len(ws_iter_index)-1 , iter_index,input_size, layerSize, ss[i]) for i in range(len(ws_iter_index))]
    print (iter_infomration   )
    """
    if True:
        ss = [0.2, 0.2, 0.3, 0.4, 0.4, 0.5]
        ks = [4, 15,50,50,50,100]
        iter_infomration = [
            calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
                                     input_size, layerSize, ss[i], pys,ks[i]) for i in range(len(ws_iter_index))]
        ks = [2, 5,10,20,100,1000]
        print (iter_infomration)
        iter_infomration = [
            calc_varitional_information(ws_iter_index[i], label, model_path, i, len(ws_iter_index) - 1, iter_index,
                                        input_size, layerSize, ss[i], pys, ks[i]) for i in range(len(ws_iter_index))]
        print (iter_infomration)
    #print (iter_infomration)
    else:
        iter_infomration = np.array(
            [processInputIter(ws_iter_index[i], bins, unique_inverse_x, unique_inverse_y, label, estimted_label, b, b1,
                              len_unique_a, pys, py_hats, pxs, py_x, i, iter_index, pys1)
             for i in range(len(ws_iter_index))])
    #ss.reverse()
    #iter_infomration= [calc_varitional_information(ws_iter_index[i], label, model_path, i,len(ws_iter_index)-1 , iter_index,input_size, layerSize, ss[i]) for i in range(len(ws_iter_index))]
    #print (iter_infomration   )
    print iter_infomration
    """
    iter_infomration = np.array(
        [processInputIter(ws_iter_index[i], bins, unique_inverse_x,unique_inverse_y, label,estimted_label,  b, b1,
                          len_unique_a, pys, py_hats, pxs, py_x, i, iter_index,pys1)
         for i in range(len(ws_iter_index))])
    """
    #print ('1 after')

    """"

    """
    #print (iter_infomration   )
    #print ('Calculated The information of epoch number - {0}'.format(iter_index))
    if True or np.mod(iter_index, int(interval_information_display)) == 0:
        print ('Calculated The information of epoch number - {0}'.format(iter_index))
    return iter_infomration

def getProb(ws, x, label,estimted_label,  num_of_bins,interval_information_display,model):
    #return np.zeros((len(ws), 2, 6, 2))
    label = np.array(label).astype(np.float)
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pys = np.sum(label, axis=0) / float(label.shape[0])
    #pys = [prob_y, 1 - prob_y]

    #prob_y_hat = [np.sum(estimted_label_local) / float(estimted_label_local.shape[1]) for estimted_label_local in estimted_label]
    #py_hats = [[local_p_y_hat, 1 - local_p_y_hat] for local_p_y_hat in prob_y_hat]
    py_hats = 0
    pxs = unique_counts / float(np.sum(unique_counts))
    bins = np.linspace(-1, 1, num_of_bins)
    py_x = []
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs,:], axis=0)
        py_x.append(py_x_current)
    py_x = np.array(py_x).T
    py_x = np.array([py_x, 1 - py_x])
    print ('Starting Calculating The information...')
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))

    if False:
        infomration = np.array(Parallel(n_jobs=NUM_CORES
                                        )(delayed(processInput)
                                                          (i,interval_information_display, ws[i], bins,unique_inverse_x,unique_inverse_y,label,estimted_label,  b, b1,len(unique_a), pys, py_hats,
                                                           pxs,py_x,pys1)
                                               for i in range( len(ws))))

    else:
        infomration = np.array([processInput
                                          (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
                                           label, estimted_label[i], b, b1, len(unique_a), pys, py_hats[i],
                                           pxs, py_x, model[i])
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
    #print ('t', layer_numer)
    PYs = np.asarray(pys).T
    from mutual_info import mutual_information
    I_est = mutual_information((data, labels[:, 0][:, None]), PYs, k=ks)
    #(I_TY / data.shape[0])
    return I_XT, I_est
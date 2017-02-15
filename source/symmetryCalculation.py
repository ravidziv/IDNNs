import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
def KL(a, b):
    D_KL = np.nansum(np.multiply(a, np.log2(np.divide(a, b))))
    return D_KL
def calculateInformation(PXs,PHgivenXs ):
    I_HX = np.sum([np.nansum(np.log2(prob_h_given_x /PHs) * (prob_h_given_x *prob_x))
                  for prob_h_given_x, prob_x in zip(PHgivenXs.T, PXs)], axis=0) 
                  
def calcSymmetry(PHgivenXs, PXgivenH, PYgivenHs, groupXs, PYX, PHs,PXs):
    unique_array, unique_indices, unique_inverse, unique_counts = \
        np.unique(groupXs, return_index=True, return_inverse=True, return_counts=True)
    DKL_YH,DKL_XH, DKL_HX = [], [],[]
    H_HgivenT,H_YgivnH=[], []
    PTs = []
    PHT = []
    PHgivenTs =[]
    for i in range(0, len(unique_array)):
        current_Xs_indexs = unique_inverse ==i
        if sum(current_Xs_indexs)>0:               
            current_PHgivenXs = PHgivenXs[:, current_Xs_indexs]
            curretnt_PXgivent = np.ones(np.sum(current_Xs_indexs))
            curretnt_PXgivent = curretnt_PXgivent / np.sum(curretnt_PXgivent)
            c_PHgivent = np.dot(current_PHgivenXs, curretnt_PXgivent)
          
            PHgivenTs.append(c_PHgivent)           
            c_PTs = np.double(np.sum(current_Xs_indexs)) / PHgivenXs.shape[1]
            PTs.append(c_PTs)
            c_PHt = np.array([c_PHgivent*c_PTs for c_PHgivent_temp in c_PHgivent.T])           
            PHT.append(c_PHt)
            current_PHgivenXs_average =np.mean(current_PHgivenXs, axis =1)
            DK_L_HX = np.array([KL(current_PHgivenXs_temp,current_PHgivenXs_average) for current_PHgivenXs_temp in current_PHgivenXs.T])
            weighted_mean_HY = np.average(DK_L_HX)            
            DKL_HX.append(weighted_mean_HY)
            index_non_zero = np.nonzero(current_PHgivenXs)
            index_non_zero_uni =  np.unique(index_non_zero[0])
            index_non_ones_uni =  np.unique(index_non_zero[1])
            current_PXgivenHs = PXgivenH[:, index_non_zero_uni]
            currentYsGivenH = PYgivenHs[:, index_non_zero_uni]     
            c_PHs = PHs[index_non_zero_uni]
            """
            #c_PHgivenT = c_PHs/np.sum(c_PHs)
            t = time.time()
            PHTs = c_PHgivent*c_PTs
            currentYH = currentYsGivenH*c_PHs          
            H_YgiveH_c = -np.nansum(PHTs*np.log2(currentYsGivenH))
            H_YgivnH.append(H_YgiveH_c)
            c_H_HgivenT = -np.nansum(PHTs*np.log2(c_PHgivent))
            H_HgivenT.append(c_H_HgivenT)
            average_prob = np.average(currentYsGivenH, axis=1,weights=c_PHs)
            DK_L_y = np.array([KL(c_YgivenH_temp,average_prob) for c_YgivenH_temp in currentYsGivenH.T])
            weighted_mean_y = np.average(DK_L_y, weights=c_PHs)
            DKL_YH.append(weighted_mean_y)
            average_prob_x = np.average(current_PXgivenHs, axis=1,weights=c_PHs)
            DK_L_x = np.array([KL(c_XgivenH_temp,average_prob_x) for c_XgivenH_temp in current_PXgivenHs.T])       
            weighted_mean_x = np.average(DK_L_x,axis=0, weights=c_PHs)    
            DKL_XH.append(weighted_mean_x)
            """
    PHgivenTs = np.array(PHgivenTs).T
    PTs = np.array(PTs)
    I_HT = np.sum([np.nansum(np.log2(prob_h_given_t /PHs) * (prob_h_given_t *prob_t))
                  for prob_h_given_t, prob_t in zip(PHgivenTs.T, PTs)], axis=0)
    
    Hht = - np.nansum(np.dot(PHgivenTs*np.log2(PHgivenTs), PTs))
    Hh = np.nansum(-PHs * np.log2(PHs))
    I_HT = Hh - Hht 
    Hhx = - np.nansum(np.dot(PHgivenXs*np.log2(PHgivenXs), PXs))
    I_HX = Hh - Hhx    
    I_HX1 = np.sum([np.nansum(np.log2(prob_h_given_x /PHs) * (prob_h_given_x *prob_x))
                  for prob_h_given_x, prob_x in zip(PHgivenXs.T, PXs)], axis=0) 
    
    I_Y
    print ('IXXXXX+++++++', I_HX1,I_HX )              
    arr_DKL_HX = np.array(DKL_HX)
    arr_DKL_YH = np.array(DKL_YH)
    arr_DKL_XH = np.array(DKL_XH)
    arr_H_HgivenT = np.array(H_HgivenT)
    arr_H_YgivnH = np.array(H_YgivnH)
    DKL_YH_mean = np.mean(arr_DKL_YH)
    DKL_XH_mean = np.mean(arr_DKL_XH)
    DKL_HX_mean = np.mean(arr_DKL_HX)
    H_HgivenT_mean = np.mean(arr_H_HgivenT)
    H_YgivnH_mean = np.mean(H_YgivnH)
    return DKL_YH_mean, DKL_XH_mean,DKL_HX_mean,H_HgivenT_mean,H_YgivnH_mean, I_HT,I_HX

def calcSymmetryOpss(PHgivenXs, PXgivenH, PYgivenHs, groupXs, PYX, PHs):
    unique_array, unique_indices, unique_inverse, unique_counts = \
        np.unique(groupXs, return_index=True, return_inverse=True, return_counts=True)
    H_TgivenH =[]
    PTs = []
    for i in range(0, PXgivenH.shape[1]):
        c_PXgivenh = PXgivenH[:,i]
        c_pTgivenh = []
        for j in range(0, len(unique_array)):
            current_Xs_indexs = unique_inverse == j
            current_Ptgivenh = np.sum(c_PXgivenh[current_Xs_indexs])
            c_pTgivenh.append(current_Ptgivenh)
        c_pTgivenh = np.array(c_pTgivenh)
        c_pTh = c_pTgivenh*PHs[i]
        c_H_TgivenH = -np.nansum(c_pTh*np.log2(c_pTgivenh))
        H_TgivenH.append(c_H_TgivenH)
    H_TgivenH = np.array(H_TgivenH)
    H_TgivenH_mean = np.average(H_TgivenH, weights=PHs)
    return H_TgivenH_mean
    
    
def calcSymmetryAll(groupXs,PTgivenXs,PYgivenTs, PYX,PTs,PXs):
    PTgivenXs = np.array(PTgivenXs)
    PYgivenTs = np.array(PYgivenTs)
    PYX = np.array(PYX)
    PTs = np.array(PTs)
    PXs = np.array(PXs)
    PXgivenTs_not_divide = np.array([np.multiply(TgivenX_temp, PXs) for TgivenX_temp in PTgivenXs]).T
    PXgivenTs = np.multiply(PXgivenTs_not_divide, np.tile((1. / (PTs)), (PTgivenXs.shape[1],1)))
    I_HT_all,I_HX_all, DKL_YGivenTs, DKL_XGivenTs,H_HgivenTs,H_TgivenHs,DKL_HGivenXs,H_YgivnHs = [], [], [], [],[],[],[],[]
    
    #print ('Beta +++++++++++++++++++++', beta)
    for ii in range(0,len(groupXs)):
        DKL_YGivenT, DKL_XGivenT,DKL_HgivenX,H_HgivenT,H_YgivnH_mean, I_HT,I_HX = calcSymmetry(PTgivenXs,PXgivenTs,  PYgivenTs, groupXs[ii],PYX,PTs,PXs)
        H_TgivenH =  calcSymmetryOpss(PTgivenXs,PXgivenTs,  PYgivenTs, groupXs[ii],PYX,PTs)
        I_HT_all.append(I_HT)
        I_HX_all.append(I_HX)
        DKL_YGivenTs.append(DKL_YGivenT)
        DKL_XGivenTs.append(DKL_XGivenT)
        DKL_HGivenXs.append(DKL_HgivenX)
        H_HgivenTs.append(H_HgivenT)
        H_TgivenHs.append(H_TgivenH)
        H_YgivnHs.append(H_YgivnH_mean)
    #print (beta,I_HT_all,I_HX_all, DKL_YGivenTs, DKL_XGivenTs,DKL_HGivenXs, H_HgivenTs,H_TgivenHs,H_YgivnHs)
    return I_HT_all,I_HX_all, DKL_YGivenTs, DKL_XGivenTs,DKL_HGivenXs, H_HgivenTs,H_TgivenHs,H_YgivnHs

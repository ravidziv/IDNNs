import numpy as np
num = 1
def KL(a, b):
    """Calculate the Kullback Leibler divergence between a and b """
    D_KL = np.nansum(np.multiply(a, np.log(np.divide(a, b+np.spacing(1)))), axis=1)
    return D_KL


def calc_information(probTgivenXs, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT

def calc_information_1(probTgivenXs, PYgivenTs, PXs, PYs, PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    #PTs = np.nansum(probTgivenXs*PXs, axis=1)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs+np.spacing(1))))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs+np.spacing(1))), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Ht - Htx
    return ITX, IYT

def calc_information(probTgivenXs, PYgivenTs, PXs, PYs, PTs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    #PTs = np.nansum(probTgivenXs*PXs, axis=1)
    t_indeces = np.nonzero(PTs)
    Ht = np.nansum(-np.dot(PTs, np.log2(PTs+np.spacing(1))))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))

    IYT = Hy - Hyt
    ITX = Ht - Htx

    return ITX, IYT


def t_calc_information(p_x_given_t, PYgivenTs, PXs, PYs):
    """Calculate the MI - I(X;T) and I(Y;T)"""
    Hx = np.nansum(-np.dot(PXs, np.log2(PXs)))
    Hxt = - np.nansum((np.dot(np.multiply(p_x_given_t, np.log2(p_x_given_t)), PXs)))
    Hyt = - np.nansum(np.dot(PYgivenTs*np.log2(PYgivenTs+np.spacing(1)), PTs))
    Hy = np.nansum(-PYs * np.log2(PYs+np.spacing(1)))
    IYT = Hy - Hyt
    ITX = Hx - Hxt
    return ITX, IYT
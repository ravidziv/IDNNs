import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from scipy import interpolate
import cPickle

def plotByEpochsEnd(I_XT_array_list, I_TY_array_list, error_array_list, train_error_list, samples, is_epoch):
    nums_arc =0

    indexs_range = I_XT_array_list[0].shape[3] if is_epoch else I_XT_array_list[0].shape[2]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, indexs_range)]
    max_num  =6
    for index, (I_XT_array, I_TY_array,error_array, train_error) in enumerate(zip(I_XT_array_list, I_TY_array_list, error_array_list, train_error_list)):
        start_num = I_XT_array.shape[4] - 2
        xs = [1]+list(range(max_num - start_num,max_num+1))
        figs, axes = [],[]
        for i in range(0, 7):
            figs.append( plt.figure())
            axes.append(figs[i].add_subplot(111))
        XT_inner,YT_inner,test_array_inner,train_array_inner = [],[],[],[]
        XT_inner_var, YT_inner_var= [],[]
        for index_in_range in range(0, indexs_range):
            XT, TY = [],[]
            XT_var, TY_var = [], []
            if is_epoch:
                train_err= np.mean(train_error[:, nums_arc,-1, index_in_range], axis=0)
                rest_err= np.mean(error_array[:, nums_arc,-1, index_in_range], axis=0)
                test_array_inner.append(rest_err)
                train_array_inner.append(train_err)
            else:
                train_err = np.mean(train_error[:, nums_arc, index_in_range,-1], axis=0)
                rest_err = np.mean(error_array[:, nums_arc, index_in_range,-1], axis=0)
                test_array_inner.append(rest_err)
                train_array_inner.append(train_err)

            for layer_index in range(0, I_XT_array.shape[4]):
                if is_epoch:
                    XT.append(np.mean(I_XT_array[:, nums_arc,-1, index_in_range, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                    XT_var.append(np.std(I_XT_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))
                    TY_var.append(np.std(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))

                else :
                    XT.append(np.mean(I_XT_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    XT_var.append(np.var(I_XT_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
                    TY_var.append(np.var(I_TY_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
            axes[0].plot(xs,XT, ':o',color= colors[index_in_range],markeredgewidth=0.0)
            axes[1].plot(xs, TY,':o',color = colors[index_in_range],markeredgewidth=0.0)
            axes[2].plot(XT, TY, ':o', color=colors[index_in_range],markeredgewidth=0.0)
            #TY[-3] +=0.0015
            ys = [0.686, 0.687, 0.691, 0.693, 0.693]
            betas = [25, 35, 51, 66, 128]
            #for ids, y in enumerate(ys):
            #    axes[0].annotate(r"$\beta=$"+str(betas[ids]), (XT[-ids-1]-0.4, y))
            #XT.append(0)
            #TY.append(0.55)
            XT_inner.append(XT)
            YT_inner.append(TY)
        x_np =np.array(XT_inner)
        yxs_np = np.array(YT_inner)
        axes[3].set_title('Test')
        axes[3].plot(XT_inner, np.tile(test_array_inner, (6, 1)).T, ':o', markeredgewidth=0.0)
        axes[4].set_title('test')
        axes[4].plot(YT_inner, np.tile(test_array_inner, (6, 1)).T, ':o', markeredgewidth=0.0)
        axes[5].set_title('Train')
        axes[5].plot(XT_inner, np.tile(train_array_inner, (6, 1)).T, ':o', markeredgewidth=0.0)
        axes[6].set_title('Train')
        axes[6].plot(YT_inner, np.tile(train_array_inner, (6, 1)).T, ':o', markeredgewidth=0.0)

        #t = np.arange(0, 1.1, .1)
        #tck = interpolate.splprep([x_np[0].tolist(), yxs_np[0].tolist()])
        #xy = np.array([[x, y] for x, y in sorted(zip(x_np[0], yxs_np[0]))])
        """"
        for l in range(0, I_XT_array.shape[4]):
            x = x_np[:,l]
            yxs = yxs_np[:, l]
            y = np.array(test_array_inner)

            axes[3].quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,color = colors
                       ,  linewidth=2, width=0.002, headwidth=2, edgecolors=colors)
            axes[4].quiver(yxs[:-1], y[:-1], yxs[1:] - yxs[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                       color=colors
                       , linewidth=2, width=0.002, headwidth=2, edgecolors=colors)

        """
        axes[0].set_ylabel('I(T;Y)')
        axes[0].set_xlabel('I(X;T)')
        ylabels = ['I(X;T)','I(T;Y)' ,'I(T;Y)','Precision','Precision']
        xlabels = ['Layer Number','Layer Number','I(X;T)','I(X;T)','I(T;Y)' ]
        ylims = [[0, 9], [0, .7],[0, .7],[0.3, 1] ,[0.3, 1] ]
        ylims = [[0,0.7]]
        for i in range(0, len(axes)):
            #axes[i].set_ylabel(ylabels[i])
            #axes[i].set_xlabel(xlabels[i])
            #axes[i].set_ylim(ylims[i])
            """"
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm._A = []
            cbar = figs[i].colorbar(sm, ticks=[0, .98])
            if is_epochs:
                cbar.ax.set_yticklabels(['0', '1000'])
                cbar.ax.set_title('Epochs')
            else:
                cbar.ax.set_yticklabels(['4%', '84%'])
                cbar.ax.set_title('Training data')
            """
            figs[i].savefig('gen_'+str(index)+'_'+str(i))

def plotMaxIB(I_XT_array_list, I_TY_array_list,I_XT_array_list_s, I_TY_array_list_s, is_epoch):
    nums_arc =0
    indexs_range = I_XT_array_list[0].shape[3] if is_epoch else I_XT_array_list[0].shape[2]
    for index, (I_XT_array_s, I_TY_array_s, I_XT_array, I_TY_array ) in enumerate(zip(I_XT_array_list_s, I_TY_array_list_s, I_XT_array_list, I_TY_array_list)):
        figs, axes = [],[]
        for i in range(0, 1):
            figs.append( plt.figure())
            axes.append(figs[i].add_subplot(111))
        XT_inner,YT_inner,test_array_inner,train_array_inner = [],[],[],[]
        for index_in_range in range(0, indexs_range):
            print (index_in_range)
            XT, TY = [],[]
            XT_var, TY_var = [], []
            XT_s, TY_s = [], []
            XT_var_s, TY_var_s = [], []
            for layer_index in range(0, I_XT_array.shape[4]):
                if is_epoch:
                    XT.append(np.mean(I_XT_array[:, nums_arc,-1, index_in_range, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                    XT_var.append(np.std(I_XT_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))
                    TY_var.append(np.std(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))
                    XT_s.append(np.mean(I_XT_array_s[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                    TY_s.append(np.mean(I_TY_array_s[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                    XT_var_s.append((np.std(I_XT_array_s[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20)))
                    TY_var_s.append((np.std(I_TY_array_s[:, nums_arc, -1, index_in_range, layer_index], axis=0))/np.sqrt(20))
                else :
                    XT.append(np.mean(I_XT_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    XT_var.append(np.var(I_XT_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
                    TY_var.append(np.var(I_TY_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
            #TY[-3] +=0.0015
            ys = [0.686, 0.687, 0.691, 0.693, 0.693]
            betas = [25, 35, 51, 66, 128]
            for ids, y in enumerate(ys):
                axes[0].annotate(r"$\beta=$"+str(betas[ids]), (XT[-ids-1]-0.4, y))
            axes[0].errorbar(XT, TY, xerr =np.array(XT_var), yerr=np.array(TY_var), color = 'g')
            axes[0].errorbar(XT_s, TY_s, xerr=np.array(XT_var_s), yerr=np.array(TY_var_s), color = 'r')
            axes[0].legend(['Network output', 'IB equations'], loc='best',numpoints = 1)
            XT_inner.append(XT)
            YT_inner.append(TY)
        axes[0].set_ylabel('I(T;Y)')
        axes[0].set_xlabel('I(X;T)')

def plotByEpoch(I_XT_array_list, I_TY_array_list,test_error_list, samples, is_epoch,range_array):
    nums_arc =0
    indexs_range = I_XT_array_list[0].shape[3] if is_epoch else I_XT_array_list[0].shape[2]
    #indexs_range = 1059
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, indexs_range)]
    max_num  =6
    range_arr = range_array[:indexs_range]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for index, (I_XT_array, I_TY_array,error_array) in enumerate(zip(I_XT_array_list, I_TY_array_list,test_error_list)):
        #ax.plot(range_arr, np.mean(error_array[:, nums_arc, -1, :indexs_range], axis=0))
        #ax1.plot(range_arr, np.mean(I_XT_array[:, nums_arc, -1, :indexs_range, -1], axis=0))
        #ax2.plot(range_arr, np.mean(I_TY_array[:, nums_arc, -1, :indexs_range, -1], axis=0))
        figs, axes = [], []
        for i in range(0, 3):
            figs.append(plt.figure())
            axes.append(figs[i].add_subplot(111))
        start_num = I_XT_array.shape[4] - 2
        xs = [1] + list(range(max_num - start_num, max_num + 1))

        XT_inner,YT_inner = [],[]
        test_array_inner = []
        layer_index = I_XT_array.shape[4]-1

        for layer_index in range(0, I_XT_array.shape[4]):
           pass

           #axes[1].plot(range_arr, np.mean(I_XT_array[:, nums_arc,-1, :indexs_range, layer_index], axis=0), linewidth = 2)
           #axes[2].plot(range_arr, np.mean(I_TY_array[:, nums_arc, -1, :indexs_range, layer_index], axis=0),linewidth = 2)
        #axes[2].plot(range_arr, np.ones(len(range_arr)) * 0.699, ':',linewidth = 2 )

        for index_in_range in range(0, indexs_range-75):
            XT, TY = [],[]
            XT_var, TY_var = [], []
            XT, TY = [], []
            XT_var, TY_var = [], []
            XT_s, TY_s = [], []
            XT_var_s, TY_var_s = [], []
            if is_epoch:
                rest_err = np.mean(error_array[:, nums_arc, -1, index_in_range], axis=0)
                test_array_inner.append(rest_err)
            else:
                rest_err = np.mean(error_array[:, nums_arc, index_in_range, -1], axis=0)
                test_array_inner.append(rest_err)

            for layer_index in range(0, I_XT_array.shape[4]):
                if is_epoch:
                    XT.append(np.mean(I_XT_array[:, nums_arc,-1, index_in_range, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                    XT_var.append(np.std(I_XT_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))
                    TY_var.append(np.std(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0)/np.sqrt(20))
                else :
                    XT.append(np.mean(I_XT_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    TY.append(np.mean(I_TY_array[:, nums_arc, index_in_range,-1, layer_index], axis=0))
                    XT_var.append(np.var(I_XT_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
                    TY_var.append(np.var(I_TY_array[:, nums_arc, index_in_range, -1, layer_index], axis=0))
            axes[0].plot(XT, TY,marker ='o', linestyle  = '-',markersize = 6 ,color=colors[index_in_range] ,markeredgewidth=0.1, linewidth = 0.2 )
            axes[1].plot(xs,XT, ':o',color= colors[index_in_range],markeredgewidth=0.0)
            axes[2].plot(xs, TY,':o',color = colors[index_in_range],markeredgewidth=0.0)
            XT_inner.append(XT)
            YT_inner.append(TY)

        axes[0].set_ylabel('I(T;Y)')
        axes[0].set_xlabel('I(X;T)')
        ylabels = ['I(T;Y)', 'I(X;T)' ,'I(T;Y)','Precision','Precision','I(X;T)','I(T;Y)']
        xlabels = ['I(X;T)', 'Layer Number','Layer Number','I(X;T)','I(T;Y)', 'Epochs', 'Epochs' ]
        ylabels = ['I(T;Y)','I(X;T)','I(T;Y)', 'error']
        xlabels = ['I(X;T)', 'Epochs', 'Epochs','Epochs' ]
        ylims = [[0, 0.71], [0, 9], [0, 0.71],[0.3, 1] ,[0.3, 1] , [0,9],[0,.7] ]
        x_np =np.array(XT_inner)
        yxs_np = np.array(YT_inner)
        """"
        for l in range(0, I_XT_array.shape[4]):
            x = x_np[:, l]
            yxs = yxs_np[:, l]
            y = np.array(test_array_inner)

            axes[3].quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                           color=colors
                           , linewidth=2, width=0.002, headwidth=2, edgecolors=colors)
            axes[4].quiver(yxs[:-1], y[:-1], yxs[1:] - yxs[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                           color=colors
                           , linewidth=2, width=0.002, headwidth=2, edgecolors=colors)
        """
        legents =  ['Layer '+str(i+1) for i in range(0, 6) ]+['H(Y)']
        for i in range(0, 3):
            axes[i].set_ylabel(ylabels[i])
            axes[i].set_xlabel(xlabels[i])
            axes[i].set_ylim(ylims[i])

            if i<1:
                axes[0].plot(range(0, 10), np.ones(10)*0.695, ':', linewidth = 2, color = 'g')
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm._A = []
                cbar = figs[i].colorbar(sm, ticks=[0, .98])
                if is_epoch:
                    cbar.ax.set_yticklabels(['0', '10000'])
                    cbar.ax.set_title('Epochs')
                else:
                    cbar.ax.set_yticklabels(['5%', '85%'])
                    cbar.ax.set_title('Training data')
            elif i==2:

                axes[i].legend(legents, loc='center left', bbox_to_anchor=(1, 0.5))
                box = axes[i].get_position()
                axes[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])
            else:
            #axes[i].legend(legents, loc='center left', bbox_to_anchor=(1, 0.5))
                box = axes[i].get_position()
                axes[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])

            figs[i].savefig('gen_'+str(index)+'_'+str(i))


def plotCompareNets(I_XT_array_list, I_TY_array_list, samples ):
    nums_arc =0
    coloros = ['red', 'blue','green','orange','yellow']
    for nums_samples in range(0, I_XT_array_list[0].shape[2]):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax1.set_title(
            'I(X;T) for different layers, samples -  ' + str(samples[nums_samples]) + '%')
        ax2.set_title(
            'I(T;Y) for different layers- samples - ' + str(samples[nums_samples]) + '%')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('I(X;T)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('I(T;Y)')
        index_net = 0
        for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
            for layer_index in range(0, I_XT_array.shape[4]):
                XT = np.mean(I_XT_array[:, nums_arc, nums_samples, :, layer_index], axis=0)
                TY = np.mean(I_TY_array[:, nums_arc, nums_samples, :, layer_index], axis=0)
                ax1.plot(XT, color= coloros[index_net])
                ax2.plot(TY,color= coloros[index_net])

            index_net += 1
        fake2Dline=[]
        names = []
        for i in range(0, len(I_XT_array_list)):
            fake2Dline.append(mpl.lines.Line2D([0], [0], linestyle="none", c=coloros[i], marker='o'))
            names.append('Net Number - ' + str(i))
        ax1.legend(fake2Dline, names, numpoints = 1, loc='best')
        ax2.legend(fake2Dline, names, numpoints = 1, loc= 'best')


def plotEpochsInfCurve(I_XT_array_list, I_TY_array_list, is_epoch):
    nums_arc = 0
    samples = [5,10, 20,30,40, 55, 70, 85]
    number = len(I_XT_array_list)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    range_index =  I_XT_array_list[0].shape[3] if is_epoch else I_XT_array_list[0].shape[2]
    for sub_name in range(1, len(I_XT_array_list)+1):
        index_net = 0
        legents = []
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
            I_XT_current = []
            I_TY_current = []
            for index_in_range in range(0,range_index):
                for layer_index in [I_XT_array.shape[4]-sub_name]:
                    if layer_index>=0:
                        if is_epoch:
                            I_XT_current.append(np.mean(I_XT_array[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                            I_TY_current.append(np.mean(I_TY_array[:, nums_arc, -1, index_in_range, layer_index], axis=0))
                        else:
                            I_XT_current.append(np.mean(I_XT_array[:, nums_arc,  index_in_range, -1,layer_index], axis=0))
                            I_TY_current.append(np.mean(I_TY_array[:, nums_arc,  index_in_range, -1, layer_index], axis=0))
                    else:
                        I_XT_current.append(np.zeros([1000]))
                        I_TY_current.append(np.zeros([ 1000]))
            if len(I_XT_current)>0:
                #legents.append('Net with ' +str(len(I_XT_array_list)-index_net) +' hidden layers')
                legents.append('Net with ' + str(samples[index_net]) +'% of the data')
                ax1.plot(I_XT_current,I_TY_current,marker='o',LineWidth=2,color=colors[index_net],markeredgewidth=0.0)
                ax2.plot(I_TY_current,marker='o',LineWidth=2,color=colors[index_net],markeredgewidth=0.0)
                ax3.plot(I_XT_current, marker='o', LineWidth=2, color=colors[index_net], markeredgewidth=0.0)


            ax1.set_xlabel('I(X;T)')
            ax1.set_ylabel('I(T;Y)')
            ax1.set_ylim([0.0, 0.7])
            ax1.set_xlim([0, 9])
            ax2.set_ylabel('I(T;Y)')
            ax2.set_ylim([0.0, 0.7])
            ax3.set_ylabel('I(X;T)')
            ax3.set_ylim([0, 9])
            index_net += 1
        ax1.legend(legents, loc='best', numpoints=1)
        ax2.legend(legents, loc='best', numpoints = 1)
        ax3.legend(legents, loc='best', numpoints=1)
        fig1.savefig('LD'+str(sub_name))
def chagneAxes(PXs,PYs, probYgivenXs, I_XH, I_HY):
    PXs = PXs.astype(np.longdouble)
    PYs = PYs.astype(np.longdouble)
    probYgivenXs = probYgivenXs.astype(np.longdouble)

    #probXgivenYs = np.vstack(probXgivenYs)
    probYgivenXs = np.array([probYgivenXs,1-probYgivenXs])
    print (PXs.shape,PYs.shape, probYgivenXs.shape)
    Hx = np.nansum(-np.dot(PXs, np.log2(PXs)))
    Hy = np.nansum(-np.dot(PYs, np.log2(PYs)))
    I_XH_normalize = (Hx -I_XH)/ Hx
    I_XH_normalize = np.log(I_XH_normalize)
    #Hxy =- np.nansum((np.dot(np.multiply(probXgivenYs, np.log2(probXgivenYs)), PYs)))
    Hyx =  - np.nansum(np.dot(probYgivenXs*np.log2(probYgivenXs), PXs))
    IYX = Hy - Hyx
    I_YH_normalize = (IYX - I_HY) / I_HY
    I_YH_normalize = np.log2(I_YH_normalize)
    return I_XH_normalize,I_YH_normalize
    
def plotNetsInfCurve(I_XT_array_list, I_TY_array_list):
    nums_arc = 0
    number = I_TY_array_list[0].shape[3]
    cmap = plt.get_cmap('gnuplot')
    num_indexs = 1200
    colors = [cmap(i) for i in np.linspace(0, 1, 1200)]
 
    x_label = '$I(X;\hat{H})$'
    y_label = '$I(Y;\hat{H})$'
    min, max = (0, I_XT_array_list[0].shape[2])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
        I_XT_current = []
        I_TY_current = []
        for layer_index in range(0, I_XT_array.shape[4]):
            I_XH = np.mean(I_XT_array[:, nums_arc, -1, :num_indexs, layer_index], axis=0)
            I_HY = np.mean(I_TY_array[:, nums_arc, -1, :num_indexs, layer_index], axis=0)
            I_XH_normalize,I_HY_normalize = chagneAxes(PXs,PYs, probYgivenXs, I_XH, I_HY)
            I_XT_current.append(I_XH_normalize)
            I_TY_current.append(I_HY_normalize)
        bew_colors  = []
        for j in range(len(colors)):
            bew_colors.append(colors[j])
            bew_colors.append(colors[j])
        plt.gca().set_color_cycle(colors)
        
        ax1.plot(np.array(I_XT_current), np.array(I_TY_current),marker ='o', linestyle  = '-',markersize = 6 ,markeredgewidth=0.1, linewidth = 0.2 )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = fig1.colorbar(sm, ticks=[0, .98])
        cbar.ax.set_yticklabels(['0', '10000'])
        cbar.ax.set_title('Epochs')
    fig1.savefig('inf_curve')

def plotSamplesInfCurve(I_XT_array_list, I_TY_array_list,error_array_list,train_error_list, samples , lim_y):
    nums_arc = 0
    legents = [str(5-i) +' layers' for i in range(0,len(I_TY_array_list))]
    number = I_TY_array_list[0].shape[2]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors', ['blue', 'red'])

    # for nums_samples in range(0, I_XT_array_list[0].shape[2]):
    index_net = 1
    min, max = (0, I_XT_array_list[0].shape[2])

    for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for nums_samples in range(0, I_XT_array_list[0].shape[2]):
            I_XT_current = []
            I_TY_current = []
            for layer_index in range(0, I_XT_array.shape[4]):
                I_XT_current.append(np.mean(I_XT_array[:, nums_arc, nums_samples, -1, layer_index], axis=0))
                I_TY_current.append(np.mean(I_TY_array[:, nums_arc, nums_samples, -1, layer_index], axis=0))
            r = (float(nums_samples) - min) / (max - min)
            g = 0
            b = 1 - r
            ax1.plot(I_XT_current,I_TY_current,':o',color=colors[nums_samples])
        ax1.set_xlabel('I(X;T)')
        ax1.set_ylabel('I(T;Y)')
        #x1.set_ylim([0.0, 0.75])
        ax1.set_xlim([0, 9])
        #ax1.legend(legents,numpoints = 1, loc='best')
        #ax1.set_title('The informatiom curve for different percent of training data')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig1.colorbar(sm, ticks=[0,1])
        cbar.ax.set_yticklabels(['4%' ,str(sampels[-1])+'%'])

def plot_by_layer_with_net(I_XT_array_list, I_TY_array_list,error_array_list,train_error_list, samples , lim_y):
    nums_arc = 0
    epchos_indexes = [10,100, 999]
    legents = [str(5-i) +' layers' for i in range(0,len(I_TY_array_list))]
    # for nums_samples in range(0, I_XT_array_list[0].shape[2]):
    for nums_samples in [I_XT_array_list[0].shape[2] - 1]:
        index_net = 1
        for epoch in epchos_indexes:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
                I_XT_current = []
                I_TY_current = []
                for layer_index in range(0, I_XT_array.shape[4]):
                    I_XT_current.append(np.mean(I_XT_array[:, nums_arc, nums_samples, epoch, layer_index], axis=0))
                    I_TY_current.append(np.mean(I_TY_array[:, nums_arc, nums_samples, epoch, layer_index], axis=0))
                ax1.plot(I_XT_current,I_TY_current,':o')
                ax1.set_xlabel('I(X;T)')
                ax1.set_ylabel('I(T;Y)')
                ax1.set_ylim([0.0, 0.75])
                ax1.set_xlim([0, 9])
                ax1.legend(legents,numpoints = 1, loc='best')
                ax1.set_title('The informatiom curve for diffrenet networks after '+str(epoch+1)+ ' epochs')
def plot_by_layer_all(I_XT_array_list, I_TY_array_list,error_array_list,train_error_list, samples , lim_y):
    nums_arc = 0
    layers_legend = ['Net with '+str(5-i)+' hidden layers' for i in range(0,I_XT_array_list[0].shape[4])]
    for nums_samples in range(0, I_XT_array_list[0].shape[2]):
    #for nums_samples in range(1,2):
        index_net = 1
        for layer_index in range(0, I_XT_array_list[0].shape[4]):
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
                if I_XT_array.shape[4]>layer_index+1:
                    XT = np.mean(I_XT_array[:, nums_arc, nums_samples, lim_y[0]:lim_y[1], layer_index], axis=0)
                    TY = np.mean(I_TY_array[:, nums_arc, nums_samples, lim_y[0]:lim_y[1], layer_index], axis=0)
                    ax1.plot(XT,linewidth=2)
                    ax2.plot(TY,linewidth=2)
            ax1.legend(layers_legend, loc='best')
            #ax1.set_ylim([1, 9])
            ax1.set_title('I(X;T) for layer number - '+str(layer_index+1)+' samples -  ' + str(samples[nums_samples]) + '%' )
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('I(X;T)')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('I(T;Y)')
            #ax2.set_ylim([0.1, 0.8])
            ax2.legend(layers_legend, loc='best')
            ax2.set_title('I(T;Y) for layer number - ' + str(layer_index+1)+' samples -  ' + str(samples[nums_samples]) + '%')
            index_net += 1
            fig1.savefig('I_XT_'+str(layer_index)+'_'+str(samples[nums_samples])+'.png')
            fig2.savefig('I_YT_'+str(layer_index) + '_' + str(samples[nums_samples]) + '.png')
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        legends_names = []
        legends_names1 =[]
        index = 1
        for error_array, train_error in zip(error_array_list, train_error_list):
            error = error_array[:, nums_arc, nums_samples, lim_y[0]:lim_y[1]]
            error = np.mean(error, axis=0)
            train_error_local = train_error[:, nums_arc, nums_samples, lim_y[0]:lim_y[1]]
            train_error_local = np.mean(train_error_local, axis=0)
            ax3.plot(error,linewidth=2)
            ax4.plot(train_error_local,linewidth=2)
            legends_names.append('Net Number- ' + str(5-index) + ' - Test')
            legends_names1.append('Net Number- ' + str(5-index) + ' - Train')
            index += 1
        ax3.set_title('Test Precision for different nets, samples - ' + str(samples[nums_samples]) + '%')
        ax3.legend(legends_names, loc='best')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Precision')
        ax4.set_title('Train Precision for different nets, samples - ' + str(samples[nums_samples]) + '%')
        ax4.legend(legends_names1, loc='best')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Precision')
        fig4.savefig('Train_'+str(layer_index) + '_' + str(samples[nums_samples]) + '.png')
        fig3.savefig('Test_'+str(layer_index)+'_'+str(samples[nums_samples])+'.png')

def plot_by_layer(I_XT_array_list, I_TY_array_list, error_array_list,train_error_list,plot_information, samples):
    nums_arc= 0
    #for nums_samples in range(0, I_XT_array_list[0].shape[2]):
    for nums_samples in [I_XT_array_list[0].shape[2]-1]:
        if plot_information:
            index_net = 1
            for I_XT_array, I_TY_array in zip(I_XT_array_list, I_TY_array_list):
                fig1,fig2 = plt.figure(), plt.figure()
                ax1, ax2 = fig1.add_subplot(111),fig1.add_subplot(111)
                for layer_index in range(0,I_XT_array.shape[4]):
                    XT = np.mean(I_XT_array[:, nums_arc, nums_samples, :, layer_index], axis=0)
                    TY = np.mean(I_TY_array[:, nums_arc, nums_samples, :, layer_index], axis=0)
                    ax1.plot(XT,linewidth= 2),ax2.plot(TY, linewidth= 2)
                layers_legend = ['Layer ' + str(i) for i in range(0, I_XT_array.shape[4])]
                leg1,leg2 = ax1.legend(layers_legend, loc='best'),ax2.legend(layers_legend,loc='best')
                ax1.set_xlabel('Epochs'),ax1.set_ylabel('I(X;T)')
                ax2.set_xlabel('Epochs'),ax2.set_ylabel('I(T;Y)')
                leg1.draggable(state=True),leg2.draggable(state=True)
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                box2 = ax2.get_position()
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.85, box2.height])
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax1.set_ylim([1, 9])
                ax2.set_ylim([0, 0.7])
                index_net +=1
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        legends_names = []
        index = 1
        for error_array, train_error in zip(error_array_list, train_error_list):
            #error =  np.mean(error_array[:, nums_arc, nums_samples, :], axis=0)
            error = error_array[:, nums_arc, nums_samples, :]
            #error = error_array[:, nums_arc, nums_samples, lim_y[0]:lim_y[1]][train_error[:, nums_arc, nums_samples, ][:,-2]>.6]
            error = np.mean(error, axis=0)
            #train_erro = np.mean(train_error[:, nums_arc, nums_samples, :], axis=0)
            train_error_local= train_error[:, nums_arc, nums_samples, :]
            #train_error_local = train_error[:, nums_arc, nums_samples, lim_y[0]:lim_y[1]][train_error[:, nums_arc, nums_samples, :][:,-2]>.6]
            train_error_local = np.mean(train_error_local, axis= 0)
            ax3.plot(error)
            #ax3.plot(train_error_local)
            legends_names.append('Net Number- '+str(index)+' - Test')
            #legends_names.append('Net Number- ' + str(index) + ' - Train')
            index +=1
        ax3.legend(legends_names, loc= 'best')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Precision')


def plotInfByEpochs(I_XT_array_list, I_TY_array_list,I_Entropy ,str_c):
    ind = np.round(2 ** np.arange(np.log2(1), np.log2(4000), .4)).astype(np.int)
    #ind = range(0,I_XT_array_list[0].shape[3])
    ind = range(0, 6000)
    ind  = range(0,5)
    for index, (I_XT_array, I_TY_array) in enumerate(zip(I_XT_array_list, I_TY_array_list )):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        #fig2 = plt.figure()
        #ax2 = fig2.add_subplot(111)
        legendts = []
        for layer_index in range(0,I_XT_array.shape[4]):
            legendts.append('Layer '+ str(layer_index+1))
            ax.plot(ind, np.mean(I_XT_array[:,0, -1, ind, layer_index], axis=0))
            ax1.plot(ind, np.mean(I_TY_array[:, 0, -1, ind, layer_index], axis=0))
            #ax2.plot(ind, np.mean(I_Entropy_array[:, 0, -1, ind, layer_index], axis=0))

        ax1.legend(legendts, loc='best')
        ax.legend(legendts, loc='best')
        #ax2.legend(legendts, loc='best')

        ax.set_ylabel('D_KL[p(x|h]||p_new(x|h)')
        ax.set_xlabel('Epochs')
        ax.set_title(str_c)
        ax1.set_ylabel('D_KL[p(y|h]||p_new(y|h)')
        ax1.set_xlabel('Epochs')
        ax1.set_title(str_c)
        #ax2.set_title(str_c)
        #ax2.set_xlabel('Epochs')
        #ax2.set_ylabel('H(h|T)')

        #ax1.set_ylabel('D_KL[p(y|t]||p_new(y|t)')
def plotError(test_error_list, train_error_list,is_epochs,epochs):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for train_error in train_error_list:
        if is_epochs:
            rest_err = np.mean(train_error[:, -1, -1, :], axis=0)
        else:
            rest_err = np.mean(train_error[:, -1, :, -1], axis=0)
        ax2.semilogy(epochs[:], rest_err)
    for ters_error in test_error_list:
        if is_epochs:
            rest_err = np.mean(ters_error[:, -1, -1, :], axis=0)
        else:
            rest_err = np.mean(ters_error[:, -1, :, -1], axis=0)
        ax2.semilogy(epochs[:],rest_err)
    #legendts = ['Net with '+str(5-i)+' hidden lsyers' for i in range(0, len(test_error_list))]
    #legendts = ['Net number ' + str(i) for i in range(0, len(test_error_list))]
    legendts = ['Train', 'Test']
    ax2.legend(legendts, loc='best')

    
#names_files = ['ts2_1_7','ts2_1_15','ts2_1_60','ts2_1_100']
#names_files = ['var_1_10','var_1_20','var_1_40','var_1_60', 'var_1_90','var_1_130', 'var_1_170','var_1_198']
#names_files = ['tf_1_198']
names_files = ['rr_1_198']
#names_files = ['ff3_1_198','ff3_2_198','ff3_3_198','ff3_4_198','ff3_5_198']

sim_TY_array_list, sim_XT_array_list, fs, I_XT_array_list,I_TY_array_list,test_error_list,train_error_list,test_error_list_loss ,train_error_list_loss=\
[],[],[],[], [], [],[],[],[]
I_XT_array_list_s, I_TY_array_list_s =[],[]
for name_s in names_files:
    curent_f = open(name_s + '.pickle', 'rb')
    d2 = cPickle.load(curent_f)
    [information_all,test_error, train_error, params2] = d2

    I_XT_array2 = information_all[:, :, :, :, :, 0]
    I_TY_array2 = information_all[:, :, :, :, :, 1]
    I_XT_array_list.append(I_XT_array2)
    I_TY_array_list.append(I_TY_array2)
    """
    strs = ['Reps - 1,2,3','Reps - 1,2','Reps - 1,3','Reps - 2,3','Reps - 1','Reps - 2','Reps - 3' ]
    for i in range(0,7):
        sim_XT_array_list, sim_TY_array_list = [], []
        I_Entropy = []
        str_c = strs[i]
        sim_XT_array2 = vars_meausre[:, :, :, :, :, i+7]
        sim_TY_array2 = vars_meausre[:, :, :, :, :, i]
        sim_XT_array_list.append(sim_XT_array2)
        sim_TY_array_list.append(sim_TY_array2)
        #I_Entropy.append(vars_meausre[:, :, :, :, :, i+14])

        #plotInfByEpochs(sim_XT_array_list, sim_TY_array_list,I_Entropy, str_c)
        #plotNetsInfCurve(sim_XT_array_list, sim_TY_array_list)
    """
    sampels = params2['samples']
    print (params2)
plotNetsInfCurve(I_XT_array_list, I_TY_array_list)
#plt.show()
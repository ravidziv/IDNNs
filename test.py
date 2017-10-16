import idnns.plots.plot_figures as plt_fig
if __name__ == '__main__':

    str_name =[['jobs/usa22_DataName=MNIST_sampleLen=1_layerSizes=400,200,100_lr=0.002_nEpochInds=22_nRepeats=1_LastEpochsInds=299_nDistSmpls=1_nEpoch=300_batch=2560/']]
    str_name = [['jobs/trails1_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=84_nRepeats=1_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4016/']]
    str_name =[['jobs/trails1_DataName=g2_sampleLen=1_layerSizes=10,7,5,4,3_lr=1e-05_nEpochInds=75_nRepeats=1_LastEpochsInds=999_nDistSmpls=1_nEpoch=1000_batch=4016/']]
    #str_name = [['jobs/trails2_DataName=var_u_sampleLen=1_layerSizes=10,7,5,4,3_lr=0.0004_nEpochInds=84_nRepeats=1_LastEpochsInds=9998_nDistSmpls=1_nEpoch=10000_batch=4016/']]
    #plt_fig.plot_figures(str_name, 2, 'd')
    plt_fig.plot_alphas(str_name[0][0])
    mode = 2
    save_name = 'figure'
    #plt_fig.plot_figures(str_name, mode, save_name)

    #plt_fig.plot_hist(str_name[0][0])
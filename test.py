import  idnns.plots.plot_figures as plt_fig
if __name__ == '__main__':

    str_name =[['jobs/usa22_DataName=MNIST_samples_len=1_layersSizes=[[400, 200, 100]]_learningRate=0.002_numEpochsInds=22_numRepeats=1_LastEpochsInds=299_num_of_disribuation_samples=1_numEphocs=300_batch=2560/']]
    str_name = [['jobs/trails1_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=84_numRepeats=1_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=4016/']]
    str_name =[['jobs/trails1_DataName=g2_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=1e-05_numEpochsInds=75_numRepeats=1_LastEpochsInds=999_num_of_disribuation_samples=1_numEphocs=1000_batch=4016/']]
    #str_name = [['jobs/trails2_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=84_numRepeats=1_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=4016/']]
    #plt_fig.plot_figures(str_name, 2, 'd')
    plt_fig.plot_alphas(str_name[0][0])
    mode = 2
    save_name = 'figure'
    #plt_fig.plot_figures(str_name, mode, save_name)

    #plt_fig.plot_hist(str_name[0][0])
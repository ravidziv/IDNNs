#IDNNs
##Description
IDNNs is a python library that implements training and calculating of information in deep neural networks [\[Shwartz-Ziv & Tishby, 2017\]](#IDNNs) in TensorFlow. The libary allows to investigate how networks look on the information plane and how it changed during the learning.

##Usage
All the code is under the 'source' directory.
For training a network and calculate the MI and the gradients of it run the main method of [run_network_with_information.py](source/run_network_with_information.py).
Off-course that you can also run only specipic methods for running only the training procedure/calculating the MI.
This file has command-line arguments as follow - 
 - 'start_samples' - The number of the first sample for calculate the information
 - 'batch_size' - The size of the batch
 - 'learning_rate' - The learning rate of the network
 - 'num_repeat' - The number of times to run the network
 - 'num_epochs' - maximum number of epochs for training
 - 'net_arch' - The architecture of the networks
 - 'per_data' - The percent of the training data
 - 'name' - The name for saving the results
 - 'data_name' - The dataset name
 - 'num_samples' - The max number of indexes for calculate the information
 - 'save_ws - True if we want to save the weights of the network
 - 'calc_information' 1 if we want to calculate the MI of the network
 - 'save_grads' - True if we want to save the gradients of the network
 - 'run_in_parallel' - True if we want to run all the networks in parallel mode
 - 'num_of_bins' - The number of bins that we divide the neurons' output
The results are save under the folder jobs. Each run create a directory with a name that contains the run properties. In this directory there are the data.pickle file with the data of run and python file that is a copy of the run_network_with_information.py that create this run.

For plotting the results run the main method in the file [plot_figures.py](source/plot_figures.py). 
This file contains methods for plotting diffret aspects of the datt (the information plane, the gradients,the norms, etc).

## References

1. <a name="IDNNs"></a> Ravid. Shwartz-Ziv, Naftali Tishby, [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810), 2017, Arxiv.

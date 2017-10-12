# IDNNs
## Description
IDNNs is a python library that implements training and calculating of information in deep neural networks
[\[Shwartz-Ziv & Tishby, 2017\]](#IDNNs) in TensorFlow. The library allows you to investigate how networks look on the
information plane and how it changes during the learning.
<img src="https://github.com/ravidziv/IDNNs/blob/master/compare_percent_mnist_5_AND_85_PERCENT_old.JPG" width="1000px"/>

## Prerequisites
- tensorflow r1.0 or higher version
- numpy 1.11.0
- matplotlib 2.0.2
- multiprocessing
- joblib

## Usage
All the code is under the `idnns/` directory.
For training a network and calculate the MI and the gradients of it run the an example in [main.py](main.py).
Off course you can also run only specific methods for running only the training procedure/calculating the MI.
This file has command-line arguments as follow - 
 - `start_samples` - The number of the first sample for calculate the information
 - `batch_size` - The size of the batch
 - `learning_rate` - The learning rate of the network
 - `num_repeat` - The number of times to run the network
 - `num_epochs` - maximum number of epochs for training
 - `net_arch` - The architecture of the networks
 - `per_data` - The percent of the training data
 - `name` - The name for saving the results
 - `data_name` - The dataset name
 - `num_samples` - The max number of indexes for calculate the information
 - `save_ws` - True if we want to save the outputs of the network
 - `calc_information` - 1 if we want to calculate the MI of the network
 - `save_grads` - True if we want to save the gradients of the network
 - `run_in_parallel` - True if we want to run all the networks in parallel mode
 - `num_of_bins` - The number of bins that we divide the neurons' output
 - `activation_function` - The activation function of the model 0 for thnh 1 for RelU'
 - `interval_accuracy_display` - The interval for display accuracy
 - `interval_information_display` - The interval for display the information calculation
 - `cov_net` - True if we want covnet
 - `rand_labels` - True if we want to set random labels
 - `data_dir` - The directory for finding the data
The results are save under the folder jobs. Each run create a directory with a name that contains the run properties. In this directory there are the data.pickle file with the data of run and python file that is a copy of the file that create this run.
The data is under the data directory. 

For plotting the results we have the file [plot_figures.py](idnns/plot/plot_figures.py). 
This file contains methods for plotting diffrent aspects of the data (the information plane, the gradients,the norms, etc).

## References

1. <a name="IDNNs"></a> Ravid. Shwartz-Ziv, Naftali Tishby, [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810), 2017, Arxiv.

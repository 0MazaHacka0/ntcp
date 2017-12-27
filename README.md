# Neural network (RNN) geneartor human-like passwords
Neural RNN network generator human passwords

# Installation
## Dependencies
1. Keras
2. numpy
3. matplotlib
4. Tensorflow-gpu
5. h5py (if you want to save your weighhts or model)
6. CuDNN and CUDA (if you want to run on GPU)
7. Python 3.6
8. TensorBoard

## Datasets
For example, in data folder you can find dataset (ICQ passwords, VK5k pass, etc)
You can use any dict, just add it to data

## Setup
###Network.py
1. In tbCallBack function change log_dir param to your log directory.
2. Change path param to your dataset folder

# Running
For running model just run it from PyCharm or from python console.

# Test
I test this model on my notebook (MSI GP72 6QF-275XRU) with
1. CPU - Intel Core i7 5700HQ
2. GPU - NVIDIA GTX960m
3. OS - Win 10 Corporate
4. Python 3.6 from Anaconda
5. PyCharm 2017.3

I use last drivers for nvidia, CUDA 8, and CuDNN for CUDA 8


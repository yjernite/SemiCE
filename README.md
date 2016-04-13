# SemiCE

Code base for semi supervised concept extraction project

## Installation

You need to have a running Python and Torch installation.

Required Python packages:
* numpy
* cPickle
* h5py

Required Torch packages:
* class
* paths
* nn
* nngraph
* rnn
* cunn
* optim
* hdf5 from [Deepmind](https://github.com/deepmind/torch-hdf5)

## Use

MakeHDF5Data/MakeHDF.py writes vocabulary and data files in hdf5 format to be read by the Torch code.

The main training script is Torch/train_models.lua.

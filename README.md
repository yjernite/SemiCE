# SemiCE

Code base for our semi supervised concept extraction project.

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

## Project organization

The different code files contain:
* CSLM.lua, ChainMRF.lua: joint probability distribution on the concepts, parametrized as a factorized or unfacrorainzed Markov chain model respectively
* Emissions.lua: conditional probability of a mention given a concept, parametrized as a multinomial or neural network
* SemiCE.lua: full generative model and recognition distribution (parametrized as a neural network)
* train_models.lua: wrapper script to initialize and train a model on unlabelled data

## Use

MakeHDF5Data/MakeHDF.py loads some labeled and unlabeled data as well as a 
dictionary mapping labels (concepts) to descriptions (mentions), and writes
vocabulary and data files in hdf5 format to be read by the Torch code.
The format for the input is described in the comments.

The main training script is Torch/train_models.lua. For information on the
arguments, run:

`$ th train_models.lua -h`

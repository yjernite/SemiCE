# SemiCE
Code base for semi supervised concept extraction project

The main training script is Torch/train_models.lua

MakeHDF5Data/MakeHDF.py writes vocab and data files in hdf5 format to be read by th Torch code

The Lua hdf5 module required is the one from [Deepmind](https://github.com/deepmind/torch-hdf5)

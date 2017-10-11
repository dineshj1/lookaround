This is code for the paper "Learning to Look Around" [https://arxiv.org/abs/1709.00507]

To see how to use it, see SUN360/example.sh.

See SUN360/config.lua to configure locations of input and output directories etc.

This repository contains a sample data file at SUN360/data/minitorchfeed/pixels_trn_torchfeed.h5 to demonstrate what an input HDF5 file should look like. For the sake of being able to run the code right out of the box, validation and test files that ship with this repository are just links to this train file. These should be overwritten when the correct files are generated.

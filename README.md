This is the original Torch code for the paper "Learning to Look Around", CVPR 2018 [https://arxiv.org/abs/1709.00507]

Note: This code no longer runs out of the box because of issues with Torch and system libraries, but we plan to release a Pytorch implementation soon. Meanwhile, this code serves only as a reference for reimplementation.

To see how to use it, see SUN360/example.sh.

See SUN360/config.lua to configure locations of input and output directories etc.

This repository contains a sample data file at SUN360/data/minitorchfeed/pixels_trn_torchfeed.h5 to demonstrate what an input HDF5 file should look like. For the sake of being able to run the code right out of the box, validation and test files that ship with this repository are just links to this train file. These should be overwritten when the correct files are generated.

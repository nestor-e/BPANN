Edward Nestor
CSCI-402 Assignment 2
BPANN - simple backpropagation neural network in C++


Compiling:
    The program van be compiled on linux systems using the g++ compiler and C++
version 11 as follows:
    g++ -std=c++11 -o NeuralNet bpann.cpp Data.cpp Network.cpp

    Also included is Makefile, which allows two versions of the program, one with
compiler optimizations enabled and on without, to be compiled by running the command:
    make

    This method will produce 2 executables, NeuralNet_Opt and NeuralNet.

Running:
    The executables can be run as follows:

    ./NeuralNet_Opt [settings file]
    or
    ./NeuralNet [settings file]

    In either case a file containing the settings to be used for this run of the
Network must be provided with the following format:

dataFiles: [Path to data file] [Type: 0=MNIST(Not implemented) , 1=mushrooms]
initialWeights: [min weight] [max weight]
learningSpeed: [initial speed]
trainingEpochs: [maximum epochs to run]
topology: [num. layer 1 nodes] [num. layer 2 nodes] ...


    The path to data file should be the full path including file name for the mushroom
data, or the path to the folder containing the 4 MNIST data files for MNIST data.

Notes:
    1) In current version, MNIST data is not supported.  Functions for loading this
        data exist in Data.cpp but are unfinished.
    2) Settings file must use Unix style line endings in order to be properly parsed.

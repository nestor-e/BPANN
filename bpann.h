/*
	Edward Nestor
	CSCI 402
	BPANN - simple back propagation neural network

	bpann.h -- contains typedefs for some common types used through out the program.
    Also contains definitions for some IO and miscelanious functions contained in
    bpann.cpp.
*/
#ifndef BPANN_H
#define BPANN_H

typedef std::tuple<std::vector<double> , std::vector<double>> Datum;
typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
void printMatrix(Matrix m);
void printVector(Vector v);

double rand_double();
void printError(const char* msg);

#endif

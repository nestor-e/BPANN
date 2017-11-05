#ifndef BPANN_H
#define BPANN_H

typedef std::tuple<std::vector<double> , std::vector<double>> Datum;
typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
void printMatrix(Matrix m);
void printVector(Vector v);

double rand_double();
void epochResults(double accuracy, std::vector<Matrix> weights);
void printError(const char* msg);

#endif

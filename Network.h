#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "bpann.h"
#include "Data.h"

class Network{
    private:
        int inputSize;
        std::vector<int> topology;
        int outputSize;
        int outputLayer;
        double trainingSpeed;

        ///  first index -> layer, second -> node in layer, third -> node in prev layer
        std::vector<Matrix> weights;

        Matrix outputs;  /// first index -> layer , second -> Node in layer
        Matrix deltas;   /// first index -> layer , second -> Node in layer

        void forward(Vector input);
        void backPropagate(Vector desired, Vector input);
        void calcDeltaHL(int layer, int node);
        void calcDeltaOL(int layer, int node, Vector* exp);
        void calcDelta(int layer, int node, Vector* exp);
        void updateWeightsHL(int layer, int node);
        void updateWeightsIL(int layer, int node, Vector* in);
        void updateWeights(int layer, int node, Vector* in);

    public:
        Network(int inputSize, std::vector<int> top, double learningSpeed);
        void randomizeWeights(double min, double max);
        void trainingEpoch(Data data);
        void testEpoch(Data data);
        bool compareResult(Vector expected);
        Vector getResult();
        static double sigmoid(double x);
        void test();

};


#endif

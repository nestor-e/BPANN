#include "Layer.h"
#include "Data.h"
#include <vector>

class Network{
    private:
        Layer* inputLayer;
        Layer* outputLayer;
        int layerCount;
        vector<int> topology;
        double trainingSpeed;
    public:
        Network(int inputSize, int outputSize, vector<int> top, double learningSpeed);
        void trainingEpoch(Data* data);
        void testEpoch(Data* data);
}

bool compareResults(Vector out, Vector exp);

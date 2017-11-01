#include "Layer.h"
#include <vector>

class Network{
    public:
        Layer* inputLayer;
        Layer* outputLayer;
        int layerCount;
        vector<int> topology;
}

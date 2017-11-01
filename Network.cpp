#include "Network.h"
#include "Layer.h"
#include "Data.h"
#include <vector>
#include <tuple>

void Network::triaingEpoch(Data* data){
    int max = data->trainingCount();
    Vector input, output;
    for(int i = 0; i < max; i++){
        std::tie(input, output) = data->trainItem(i);
        inputLayer->forward(input);
        outputLayer->backPropagateDelta(output);
        inputLayer->updateWeights(trainingSpeed);
    }
}

void Network::testEpoch(Data* data){
    int max = data->testCount();
    Vector input, output;
    for(int i = 0; i < max; i++){
        std::tie(input, output) = data->testItem(i);
        inputLayer->forward(input);
        compareResults(outputLayer->getResults(), output);
    }
}

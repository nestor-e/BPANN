#include "Network.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <cassert>
#include <iostream>

Network::Network(int inputSize, std::vector<int> top, double learningSpeed){
    topology = top;
    trainingSpeed = learningSpeed;
    int layers = topology.size();
    this->inputSize = inputSize;
    outputSize = topology[layers-1];
    outputLayer = layers - 1;

    // Initialize all arrays to correct size
    weights.resize(layers);
    outputs.resize(layers);
    deltas.resize(layers);
    for(int i = 0; i < layers; i++){
        outputs[i].resize(topology[i]);
        deltas[i].resize(topology[i]);
        weights[i].resize(topology[i]);
        int prevSize = (i > 0)?(topology[i-1]):(inputSize);
        for(int j = 0; j < topology[i]; j++){
            weights[i][j].resize(prevSize);
        }
    }
}

void Network::randomizeWeights(double min, double max){
    double range = max - min;
    int layers = topology.size();
    for(int i = 0; i < layers; i++){
        int nodes = topology[i];
        for(int j = 0; j < nodes; j++){
            int prevNodes = weights[i][j].size();
            for(int k = 0; k < prevNodes; k++){
                weights[i][j][k] = min + ( range*rand_double() );
            }
        }
    }
}


void Network::forward(Vector input){
    assert(input.size() == inputSize);
    assert(topology.size() > 0);
    int nodes = topology[0];
    int layers = topology.size();
    int prevNodes;
    double temp;
    // First layer is diffrent since it has a seperate input
    for(int j = 0; j < nodes; j++){
        temp = 0.0;
        for(int k = 0; k < inputSize; k++){
            temp += weights[0][j][k] * input[k];
        }
        outputs[0][j] = sigmoid(temp);
    }

    for(int i = 1; i < layers; i++){
        prevNodes = nodes;
        nodes = topology[i];
        for(int j = 0; j < nodes; j++){
            temp = 0;
            for(int k = 0; k < prevNodes; k++){
                temp += weights[i][j][k] * outputs[i-1][k];
            }
            outputs[i][j] = sigmoid(temp);
        }
    }
}

void Network::calcDeltaOL(int layer, int node, Vector* exp){
    double thisOut = outputs[layer][node];
    deltas[layer][node] = thisOut * (1 - thisOut) * ((*exp)[node] - thisOut);
}

void Network::calcDeltaHL(int layer, int node){
    int oLayer = layer +  1;
    int oSize = topology[oLayer];
    double thisOut = outputs[layer][node];
    double dSum = 0.0;
    for(int i = 0; i < oSize; i++){
        dSum += weights[oLayer][i][node] * deltas[oLayer][i];
    }
    deltas[layer][node] = thisOut  * (1 - thisOut) * dSum;
}


void Network::calcDelta(int layer, int node, Vector* ex){
    if(layer >= outputLayer){
        calcDeltaOL(layer, node, ex);
    } else {
        calcDeltaHL( layer, node);
    }
}

void Network::updateWeightsIL(int layer, int node, Vector* in){
    int oSize = in->size();
    double nDelta = deltas[layer][node] * trainingSpeed;
    for(int i = 0; i < oSize; i++){
        weights[layer][node][i] += nDelta * (*in)[i];
    }
}

void Network::updateWeightsHL(int layer, int node){
    int oLayer = layer - 1;
    int oSize =  topology[oLayer];
    double nDelta = deltas[layer][node] * trainingSpeed;
    for(int i = 0; i < oSize; i++){
        weights[layer][node][i] += nDelta * outputs[oLayer][i];
    }
}

void Network::updateWeights(int layer, int node, Vector* in){
    if(layer <= 0){
        updateWeightsIL( layer, node, in);
    } else {
        updateWeightsHL( layer, node);
    }
}

void Network::backPropagate(Vector expected, Vector input){
    assert(expected.size() == outputSize);
    assert(topology.size() > 0);

    for(int layer = outputLayer; layer >= 0; layer--){
        int nodeCount = topology[layer];
        for(int node = 0; node < nodeCount ; node++){
            calcDelta(layer, node, &expected);
        }
    }

    for(int layer = outputLayer; layer >= 0; layer--){
        int nodeCount = topology[layer];
        for(int node = 0; node < nodeCount ; node++){
            updateWeights(layer, node, &input);
        }
    }
}

Vector Network::getResult(){
    Vector r (outputSize);
    for(int i = 0; i < outputSize; i++){
        r[i] = (outputs[outputLayer][i] > 0.5)?(1):(0);
    }
    return r;
}

bool Network::compareResult(Vector expected){
    bool match = true;
    double temp;
    for(int  i = 0; i < outputSize; i++){
        temp = (outputs[outputLayer][i]  > 0.5)?(1):(0);
        if(temp != expected[i]){
            match = false;
        }
    }
    return match;
}

void Network::testEpoch(Data data){
    if(topology.size() <= 0){
        return;
    }

    int items = data.testCount();
    int correct = 0;
    for(int i = 0; i < items; i++){
        Datum d = data.testItem(i);
        Vector in = std::get<0>(d);
        Vector out = std::get<1>(d);
        if(in.size() == inputSize && out.size() == outputSize){
            forward(in);
            if(compareResult(out)){
                correct++;
            }
        } else {
            printError("Mismatched size training example found.");
        }
    }
    double accuracy = ((double) correct) / items;
    epochResults(accuracy, weights);
}

void Network::trainingEpoch(Data data){
    if(topology.size() <= 0){
        return;
    }

    int items = data.trainCount();
    int correct = 0;
    for(int i = 0; i < items; i++){
        Datum d = data.trainItem(i);
        Vector in = std::get<0>(d);
        Vector out = std::get<1>(d);
        if(in.size() == inputSize && out.size() == outputSize){
            forward(in);
            if(compareResult(out)){
                correct++;
            }
            backPropagate(out, in);
        } else {
            printError("Mismatched size training example found.");
        }
    }
    double accuracy = ((double) correct) / items;
    epochResults(accuracy, weights);
}


double Network::sigmoid(double x){
    return 1.0 / (1.0 + exp(-1.0 * x));
}


void Network::test(){
    Vector in = {1, 0, 1};
    Vector out = {1};
    weights[0][0][0] = 0.25;
    weights[0][0][1] = -0.3;
    weights[0][0][2] = -0.1;
    weights[0][1][0] = 0.5;
    weights[0][1][1] = 0.8;
    weights[0][1][2] = -0.2;
    weights[1][0][0] = 0.5;
    weights[1][0][1] = 0.4;
    forward(in);
    std::cout << "Forward Step:" << std::endl;
    std::cout << "\tInput:"<< std::endl;
    printVector(in);
    std::cout << "\tL1: " << std::endl;
    std::cout << "\t\tOutput: " << std::endl;
    printVector(outputs[0]);
    std::cout << "\t\tWeights: " << std::endl;
    printMatrix(weights[0]);
    std::cout << "\tL2:" << std::endl;
    std::cout << "\t\tOutput: " << std::endl;
    printVector(outputs[1]);
    std::cout << "\t\tWeights: " << std::endl;
    printMatrix(weights[1]);

    std::cout << std::endl<< std::endl << "After Back:" << std::endl;
    backPropagate(out, in);
    std::cout << "\tL1: " << std::endl;
    std::cout << "\t\tOutput: " << std::endl;
    printVector(outputs[0]);
    std::cout << "\t\tDelta: " << std::endl;
    printVector(deltas[0]);
    std::cout << "\t\tWeights: " << std::endl;
    printMatrix(weights[0]);
    std::cout << "\tL2:" << std::endl;
    std::cout << "\t\tOutput: " << std::endl;
    printVector(outputs[1]);
    std::cout << "\t\tDelta: " << std::endl;
    printVector(deltas[1]);
    std::cout << "\t\tWeights: " << std::endl;
    printMatrix(weights[1]);
}

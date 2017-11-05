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

void Network::backPropagate(Vector expected, Vector input){
    assert(expected.size() == outputSize);
    assert(topology.size() > 0);

    double temp;
    int lSize, prevSize, nextSize;
    lSize = topology[outputLayer];
    prevSize = (outputLayer > 0)?(topology[outputLayer - 1]):(inputSize);
    for(int j = 0; j < lSize; j++){
        temp = outputs[outputLayer][j];
        temp = temp * (1 - temp) * (expected[j] - temp);
        deltas[outputLayer][j] = temp;
        temp = temp * trainingSpeed;
        for(int k = 0; k < prevSize; k++){
            weights[outputLayer][j][k] += temp;
            if(outputLayer > 0){
                temp *= outputs[outputLayer - 1][k];
            } else {
                temp *= input[k];
            }
        }
    }

    for(int i = outputLayer - 1; i >= 0; i--){
        nextSize = lSize;
        lSize = prevSize;
        prevSize = (i > 0)?(topology[i - 1]):(inputSize);
        for(int j = 0; j < lSize; j++){
            temp = 0;
            for(int k = 0; k < nextSize; k++){
                temp += weights[i + 1][k][j] * deltas[i + 1][k];
            }
            temp *= outputs[i][j] * (1 - outputs[i][j]);
            deltas[i][j] = temp;
            temp = temp * trainingSpeed;
            for(int k = 0; k < prevSize; k++){
                if(i > 0){
                    temp *= outputs[i - 1][k];
                } else {
                    temp *= input[k];
                }
                weights[i][j][k] += temp;
            }
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

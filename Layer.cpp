#include "Layer.h"
#include <vector>
#include <cassert>
#include <cmath>

double randDouble(){
    return ((double) rand()) / RAND_MAX;
}

Layer::Layer(int outSize, int inSize, bool useBias){
    prev = NULL;
    next = NULL;
    outputSize = outSize;
    inputSize = inSize;
    this->useBias = useBias;
}

Layer::Layer(int size; Layer* prev, bool useBias){
	this->prev = prev;
    prev->next = this;
	next = NULL;
	outputSize = size;
	inputSize = prev.outputSize;
	this->useBias = useBias;
}

Layer::randomInit(){
    weights.reserve(outputSize);
    beta.reserve(outputSize);
    bias.reserve(outputSize);
    out.reserve(outputSize);
    delta.reserve(outputSize);
    for(int i = 0; i < outputSize; i++){
        Vector w (inputSize);
        for(int j = 0; j < inputSize; j++){
            w[j] = randDouble();
        }
        weights[i] = w;
        bias[i] = randDouble();
        beta[i] = 1.0;
        out[i] = 0.0;
        delta[i] = 0.0;
    }
}

Layer::forward(Vector input){
    assert(prev == NULL);
    assert(input.size() == inputSize);
    for(int i = 0 ; i < outputSize; i++){
        double temp = 0;
        for(int j = 0; j < inputSize; j++){
            temp += input[j]*weights[i][j];
        }
        if(useBias){
            temp += bias[i];
        }
        out[i] = sigmoid(temp, i);
    }
    if(next != NULL){
        next->forward();
    }
}

Layer::forward(){
    assert(prev != NULL);
    for(int i = 0 ; i < outputSize; i++){
        double temp = 0;
        for(int j = 0; j < inputSize; j++){
            temp += prev->out[j]*weights[i][j];
        }
        if(useBias){
            temp += bias[i];
        }
        out[i] = sigmoid(temp, i);
    }
    if(next != NULL){
        next->forward();
    }
}

Layer::backPropagateDelta(Vector expected){
    assert(next==NULL);
    assert(expected.size() == outputSize);
    for(int i = 0; i < outputSize; i++){
        delta[i] = out[i] * (1.0 - out[i]) * (expected[i] - out[i]);
    }
    if(prev != NULL){
        prev->backPropagateDelta();
    }
}

Layer::backPropagateDelta(){
    assert(next!=NULL);
    for(int i = 0; i < outputSize; i++){
        delta[i] = out[i] * (1.0 - out[i]);
        double temp = 0;
        for(int j = 0; j < next->outputSize; j++){
            temp += next->weights[j][i] * next->delta[j];
        }
        delta[i] *= temp;
    }
    if(prev != NULL){
        prev->backPropagateDelta();
    }
}


Layer::updateWeights(double speed){
    for(int i = 0 ; i < outputSize; i++){
        for(int j = 0; j < inputSize; j++){
            weights[i][j] += speed * delta[i];
        }
    }
    if(next != NULL){
        next->updateWeights(speed);
    }
}


Layer::sigmoid(double t, int i){
    return 1.0 / (1.0 + exp(-1.0 * beta[i] * t));
}

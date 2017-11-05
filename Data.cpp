#include "Data.h"
#include <fstream>
#include <iostream>
#include <string>

const double Data::translationTable[22][26] = {
//   a b c d e f g h i j k l m n o p q r s t u v w x y z
    {0,1,2,0,0,4,0,0,0,0,5,0,0,0,0,0,0,0,6,0,0,0,0,3,0,0},
    {0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,3,0},
    {0,2,3,0,8,0,4,0,0,0,0,0,0,1,0,6,0,5,0,0,7,0,9,0,10,0},
    {0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {1,0,3,0,0,5,0,0,0,0,0,2,6,7,0,8,0,0,9,0,0,0,0,0,4,0},
    {1,0,0,2,0,3,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,3,0,0,10,0,5,4,0,0,1,0,0,2,7,8,0,6,0,0,9,0,11,0,12,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0},
    {0,1,2,0,4,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,3,0,0,0,0,5},
    {0,0,0,0,0,1,0,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0,0,0,2,0},
    {0,0,0,0,0,1,0,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0,0,0,2,0},
    {0,2,3,0,7,0,4,0,0,0,0,0,0,1,5,6,0,0,0,0,0,0,8,0,9,0},
    {0,2,3,0,7,0,4,0,0,0,0,0,0,1,5,6,0,0,0,0,0,0,8,0,9,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,3,0,4,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,3,0,0,0,0,0,0},
    {0,0,1,0,2,3,0,0,0,0,0,4,0,5,0,6,0,0,7,0,0,0,0,0,0,8},
    {0,3,0,0,0,0,0,4,0,0,1,0,0,2,6,0,0,5,0,0,7,0,8,0,9,0},
    {1,0,2,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,4,0,0,5,0,0,6,0},
    {0,0,0,7,0,0,1,0,0,0,0,2,3,0,0,4,0,0,0,0,5,0,6,0,0,0}
};

Data::Data(int type){
    this->type = type;
    ready = false;
}

bool Data::init(char* fileName){
    bool s;
    switch(type){
        case 0:
            s = initMINST(fileName);
            break;
        case 1:
            s = initMush(fileName);
            break;
        default:
            s = false;
    }
    ready = s;
    return s;
}

bool Data::initMINST(char* filename){
	return false;
}

bool Data::initMush(char* fileName){
    std::ifstream file;
    file.open(fileName);
    if(!file.is_open()){
        std::cerr << "Unable to open " << fileName << " for reading." << std::endl;
        return false;
    }
    std::string line;
    while(std::getline(file, line)){
        if(line.length() >= 45){
            Vector ex (1);
            Vector in (22);
            ex[0] = (line[0] == 'e')?(1.0):(0.0);
            for(int i = 0; i < 22; i++){
	                in[i] = mushroomTranslation(i, line[2*i + 2]);
            }
            Datum d = std::make_tuple(in, ex);
            if(rand_double() <= TEST_POP_RATE){
                testValues.push_back(d);
            } else {
                trainingValues.push_back(d);
            }
        }
    }
    return true;
}

double Data::mushroomTranslation(int item, char value){
    int cVal = (int)(value - 'a');
    if(item >= 0 && item < 22 && cVal >= 0 && cVal < 26){
        return translationTable[item][cVal];
    } else {
        return 0;
    }
}


int Data::testCount(){
    return testValues.size();
}

Datum Data::testItem(int idx){
    return testValues[idx % testValues.size()];
}

int Data::trainCount(){
    return trainingValues.size();
}

Datum Data::trainItem(int idx){
    return trainingValues[idx % trainingValues.size()];
}


int Data::getInputSize(){
    if(trainingValues.size() > 0){
        Vector in = std::get<0>(trainingValues[0]);
        return in.size();
    } else {
        return 0;
    }
}

int Data::getOutputSize(){
    if(trainingValues.size() > 0){
        Vector out = std::get<1>(trainingValues[0]);
        return out.size();
    } else {
        return 0;
    }
}

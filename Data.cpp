#include "Data.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>

#define TEST_POP_RATE 0.15


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

bool Data::initMush(char* fileName){
    ifstream file;
    file.open(fileName);
    if(!file.isOpen()){
        cerr << "Unable to open " << fileName << " for reading." << endl;
        return false;
    }
    string line;
    while(std::getline(file, line)){
        if(line.lenght() >= 45){
            Vector ex (1);
            Vector in (22);
            ex[0] = line[0] == 'e';
            for(int i = 0; i < 22; i++){
                in[i] = mushroomTranslationTable(i, line[2*i + 2]);
            }
            Datum d = std::make_tuple(in, ex);
            if(rand_double() <= TEST_POP_RATE){
                testValues.push_bacK(d);
            } else {
                trainingEpoch.push_back(d);
            }
        }
    }
    return true;
}

int Data::mushroomTranslation(int item, char value){
    int cVal = (int)(value - 'a');
    if(item >= 0 && item < 22 && cVal >= 0 && cVal < 26){
        return translationTable[item][cVal];
    } else {
        return 0;
    }
}

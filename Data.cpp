#include "Data.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>

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

}

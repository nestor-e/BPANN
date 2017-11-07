/*
	Edward Nestor
	CSCI 402
	BPANN - simple back propagation neural network

	Data.cpp -- contains code for parsing mushrooms.csv file and creating an object
    to represent the data points it contains.  Interfaces with rest of program are
    generic enough to allow support for other datasets (such as MNIST hadwritting
    recognition data) to be added, but none are in place yet.
*/
#include "Data.h"
#include <fstream>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>


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
            s = initMNIST(fileName);
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


double Data::valToDouble(unsigned char x){
    return ((double) x) / 255;
}
Vector Data::valTo1Of10(unsigned char x){
    std::vector<double> temp (10, 0);
    int i = (int) x;
    if(i < 0 || i > 9){
        std::cerr << "Data parse error, output out of range" << std::endl;
    } else {
        temp[i] = 1.0;
    }

    return temp;
}

    //TODO: Fix this, slow and leaks memeory
void Data::parseMinstData(void* lMap, void* iMap, bool isTrain){
    // int items = (isTrain)?(60000):(10000);
    // int rows = 28;
    // int cols = 28;
    //
    // int lOffset = 8;
    // int iOffset = 16;
    //
    // unsigned char* lStart = (unsigned char*) lMap;
    // unsigned char* iStart = (unsigned char*) iMap;
    // std::vector<Datum> tempData (items);
    // for(int i = 0; i < items; i++){
    //     if(i % 2500 == 0){
    //         std::cout << i << " / " << items << std::endl;
    //     }
    //     Vector input (rows*cols);
    //     Vector output = valTo1Of10(lStart[i + lOffset]);
    //     for(int r = 0; r < rows; r++){
    //         for(int c = 0; c < cols; c++){
    //             input.push_back(valToDouble( iStart[iOffset] ));
    //             iOffset++;
    //         }
    //     }
    //     Datum d = std::make_tuple(input, output);
    //     tempData.push_back(d);
    //     if(isTrain){
    //         trainingValues = tempData;
    //     } else {
    //         testValues = tempData;
    //     }
    //
    // }
}


bool Data::initMNIST(char* filename){
    return false;
 //    // MNIST parsing not currently working
 //    int bufSize = strlen(filename) + 64;
 //    char buf[bufSize];
 //    char* testLabels = (char*) "t10k-labels.idx1-ubyte";
 //    char* testImgs = (char*) "t10k-images.idx3-ubyte";
 //    char* trainLabels = (char*) "train-labels.idx1-ubyte";
 //    char* trainImgs = (char*) "train-images.idx3-ubyte";
 //
 //    strncpy(buf, filename, bufSize);
 //    strcat(buf, testImgs);
 //    int testImgFd = open(buf, O_RDONLY);
 //    strncpy(buf, filename, bufSize);
 //    strcat(buf, testLabels);
 //    int testLabelFd = open(buf, O_RDONLY);
 //
 //    strncpy(buf, filename, bufSize);
 //    strcat(buf, trainImgs);
 //    int trainImgFd = open(buf, O_RDONLY);
 //    strncpy(buf, filename, bufSize);
 //    strcat(buf, trainLabels);
 //    int trainLabelFd = open(buf, O_RDONLY);
 //
 //    if(testImgFd < 0 || testLabelFd < 0 || trainImgFd < 0 || trainLabelFd < 0){
 //        std::cerr << "Unable to open files" << std::endl;
 //        close(testImgFd);
 //        close(testLabelFd);
 //        close(trainImgFd);
 //        close(trainLabelFd);
 //        return false;
 //    }
 //
 //    struct stat stats;
 //    fstat(testImgFd, &stats);
 //    size_t testImgSize = stats.st_size;
 //    fstat(testLabelFd, &stats);
 //    size_t testLabelSize = stats.st_size;
 //    fstat(trainImgFd, &stats);
 //    size_t trainImgSize = stats.st_size;
 //    fstat(trainLabelFd, &stats);
 //    size_t trainLabelSize = stats.st_size;
 //
 //    // Load Training data
 //    std::cout << "Loading training data:" << std::endl;
 //    void* labels = mmap(NULL, trainLabelSize, PROT_READ, MAP_PRIVATE, trainLabelFd, 0);
 //    void* imgs = mmap(NULL, trainImgSize, PROT_READ, MAP_PRIVATE, trainImgFd, 0);
 //
 //    if(labels == MAP_FAILED || imgs == MAP_FAILED){
 //        std::cerr << "File Mapping failed" << std::endl;
 //        munmap(labels, trainLabelSize);
 //        munmap(imgs, trainImgSize);
 //        close(testImgFd);
 //        close(testLabelFd);
 //        close(trainImgFd);
 //        close(trainLabelFd);
 //        return false;
 //    }
 //
 //    parseMinstData(labels, imgs, true);
 //    munmap(labels, trainLabelSize);
 //    munmap(imgs, trainImgSize);
 // // Load Test data
 //    std::cout << "Loading test data:" << std::endl;
 //    labels = mmap(NULL, testLabelSize, PROT_READ, MAP_PRIVATE, testLabelFd, 0);
 //    imgs = mmap(NULL, testImgSize, PROT_READ, MAP_PRIVATE, testImgFd, 0);
 //
 //    if(labels == MAP_FAILED || imgs == MAP_FAILED){
 //        std::cerr << "File Mapping failed" << std::endl;
 //        munmap(labels, testLabelSize);
 //        munmap(imgs, testImgSize);
 //        close(testImgFd);
 //        close(testLabelFd);
 //        close(trainImgFd);
 //        close(trainLabelFd);
 //        return false;
 //    }
 //
 //    parseMinstData(labels, imgs, false);
 //    munmap(labels, testLabelSize);
 //    munmap(imgs, testImgSize);
 //    close(testImgFd);
 //    close(testLabelFd);
 //    close(trainImgFd);
 //    close(trainLabelFd);
 //    return true;
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

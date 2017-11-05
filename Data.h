#ifndef DATA_H
#define DATA_H

#include <vector>
#include <tuple>
#include "bpann.h"

#define TEST_POP_RATE 0.15

class Data{
    private:
        int type; // 0 = minst, 1 = Mushrooms
        bool ready;
        std::vector< Datum > trainingValues;
        std::vector< Datum > testValues;
        bool initMINST(char* fileName);
        bool initMush(char* fileName);
        static const double translationTable[22][26];
        static double mushroomTranslation(int item, char value);
    public:
        Data(int type); // 0 = minst, 1 = Mushrooms
        bool init(char* fileName);
        Datum testItem(int idx);
        Datum trainItem(int idx);
        int testCount();
        int trainCount();
        int getInputSize();
        int getOutputSize();
};


#endif

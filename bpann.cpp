#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "Layer.h"
#include "Network.h"


int main(int argc, char* argv[]){
		const int x = atoi(argv[1]);
		vector<int> a1(x);
		for(int i = 0; i < a1.size(); i++){
			a1.at(i)= i;
		}
		for(int i = 0; i < a1.size(); i++){
			cout <<  a1.at(i) << endl;
		}
    return 0;
}

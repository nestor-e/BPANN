/*
	Edward Nestor
	CSCI 402
	BPANN - simple back propagation neural network

	bpann.cpp -- contains entry point main(), handles cmd-line argument parsing,
	as well as most of the console IO for the program, and random nuber generation.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "bpann.h"
#include "Data.h"
#include "Network.h"

using namespace std;

struct ProgramSettings  {
	int fileType;
	int rounds;
	double minWeight;
	double maxWeight;
	double speed;
	char* fileName;
	vector<int>* topology;
};

bool readSettings(char* fileName, ProgramSettings* settings){
	ifstream sFile;
	sFile.open(fileName);
	if(!sFile.is_open()){
		cerr << "Could not open file " << fileName << endl;
		return false;
	}
	vector<int>* levels = new vector<int>;
	string line, label, dataFile;
	int fileType, rounds, n;
	double min, max, speed;
	if(getline(sFile, line)){
		stringstream ls (line);
		ls >> label >> dataFile >> fileType;
	} else {
		return false;
	}
	if(getline(sFile, line)){
		stringstream ls (line);
		ls >> label >> min >> max;
	} else {
		return false;
	}
	if(getline(sFile, line)){
		stringstream ls (line);
		ls >> label >> speed;
	} else {
		return false;
	}
	if(getline(sFile, line)){
		stringstream ls (line);
		ls >> label >> rounds;
	} else {
		return false;
	}
	if(getline(sFile, line)){
		stringstream ls (line);
		ls >> label;
		while(ls >> n){
			levels->push_back(n);
		}
	} else {
		return false;
	}

	if(rounds < 1 || speed < 0 || levels->size() < 1 || dataFile.length() < 1){
		return false;
	}

	settings->rounds = rounds;
	settings->speed = speed;
	settings->minWeight = min;
	settings->maxWeight = max;
	settings->topology = levels;
	settings->fileName = new char[dataFile.length() + 1];
	settings->fileType = fileType;
	for(int i = 0; i < dataFile.length(); i++){
		settings->fileName[i] = dataFile[i];
	}
	settings->fileName[dataFile.length()] = '\0';
	return true;
}


double rand_double(){
	return ((double) rand()) / RAND_MAX;
}

string vectorToString(Vector v){
	int s = v.size();
	if(s <= 0){
		return "< >";
	}
	stringstream sb;
	sb.setf(ios::fixed, ios::floatfield);
	sb << setprecision(5) << "< "<< setw(8) << v[0];
	for(int i = 1; i < s;  i++){
		sb << " , " << setw(8) << v[i];
	}
	sb << " >";
	return sb.str();
}

string matrixToString(Matrix m){
	int r = m.size();
	if(r <= 0) {
		return "[ ]";
	}
	stringstream sb;
	sb << "[   " << vectorToString(m[0]) ;
	for(int i = 1; i < r; i++){
		sb << endl << "    " << vectorToString(m[i]);
	}
	sb << "   ]";
	return sb.str();
}

void printVector(Vector v){
	cout << vectorToString(v) << endl;
}


void printMatrix(Matrix m){
	cout << matrixToString(m) << endl;
}

void printUsage(){
	cerr << "Usage: ./BpNetwork {settings file}" << endl;
	cerr << "Settings file format:" << endl;
	cerr << "dataFiles: {Data file path} {Type: 1 - mushrooms, 0 - MNIST(Not Implemented)}" << endl;
	cerr << "initialWeights: {minWeight} {maxWeight}" << endl;
	cerr << "learningSpeed: {spped}" << endl;
	cerr << "trainingEpochs: {max rounds}" << endl;
	cerr << "topology: {L1} {L2} {L3} ..." << endl;
}

void printError(const char* msg){
	cerr << msg << endl;
}

int main(int argc, char* argv[]){

	if(argc != 2){
		printUsage();
		return 1;
	}

	ProgramSettings ps;
	if( !readSettings(argv[1], &ps)){
		printUsage();
		return 1;
	}

	srand(time(NULL));


	Data d (ps.fileType);
	if( !d.init(ps.fileName) ){
		cerr << "Failed to parse " << ps.fileName << endl;
		return 1;
	}
	if(d.getOutputSize() != ps.topology->back()){
		cerr << "Topology incompatible with data." << endl;
		return 1;
	}

	Network nw (d.getInputSize(), *(ps.topology), ps.speed);
	nw.randomizeWeights(ps.minWeight, ps.maxWeight);

	double acc, lastAcc = 0, bestAcc = 0;
	bool improveStep;
	cout << "Test topology:";
	for(int i = 0; i < ps.topology->size(); i++){
		cout << " " << (*ps.topology)[i];
	}
	cout << endl << "Learning speed: " << ps.speed << endl;
	cout << "Initial weights: " << endl;
	nw.printWeights();
	cout << "------------------------------------------------------------------";
	cout << endl << endl;
	clock_t startTime = clock();
	for(int i = 1 ; i <= ps.rounds; i++){
		acc = nw.trainingEpoch(d);
		bestAcc = (acc > bestAcc)?(acc):(bestAcc);
		improveStep = (1 - acc) < (1 - lastAcc) / 2;
		if(i % 10 == 0 || improveStep){
			cout << "Training Epoch " << i << " completed with ";
			cout << fixed << setprecision(2) << acc*100 << "% accuracy." << endl;
			if(improveStep){
				lastAcc = acc;
				nw.printWeights();
			}
		}
	}
	cout << "------------------------------------------------------------------";
	cout << endl << endl;
	double trainingTime = (double)(clock() - startTime)/CLOCKS_PER_SEC;
	cout << "Training completed in ~" << setprecision(2) << trainingTime << " sec." << endl;
	cout << "Final weights:" << endl;
	nw.printWeights();
	acc = nw.testEpoch(d);
	cout << "Test Epoch completed with ";
	cout << fixed << setprecision(2) << acc*100 << "% accuracy." << endl;

	return 0;
}

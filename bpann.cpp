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
	int trainingRounds;
	int fileType;
	int rounds;
	double minWeight;
	double maxWeight;
	double speed;
	char* fileName;
	vector<int>* topology;
};

struct epochStatus_st {
	int count;
	bool inTraining;
	double bestAccuracy;
	double lastAccuracy;
} epochStatus;

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
	cerr << "dataFiles: {Data file path} {Type: 1 - mushrooms, 2 - MINST(not implemented)}" << endl;
	cerr << "initialWeights: {minWeight} {maxWeight}" << endl;
	cerr << "learningSpeed: {spped}" << endl;
	cerr << "trainingEpochs: {max rounds}" << endl;
	cerr << "topology: {L1} {L2} {L3}" << endl;
}

void epochResults(double accuracy, vector<Matrix> weights){
	epochStatus.count++;
	if(accuracy > epochStatus.bestAccuracy){
		epochStatus.bestAccuracy = accuracy;
	}
	cout << "Epoch " << epochStatus.count << " completed with ";
	cout << fixed << setprecision(2) << accuracy*100 << "% accuracy." << endl;
	if(!epochStatus.inTraining || accuracy > (epochStatus.lastAccuracy * 1.1)){
		epochStatus.lastAccuracy = accuracy;
		cout << "Weights:" << endl;
		for(int i = 0 ; i < weights.size(); i++){
			cout << "Layer " << i + 1 << ":" << endl;
			printMatrix(weights[i]);
		}
	}
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
	epochStatus.count = 0;
	epochStatus.inTraining = true;
	epochStatus.bestAccuracy = 0;
	epochStatus.lastAccuracy = 0;



	Data d (ps.fileType);
	if( !d.init(ps.fileName) ){
		cerr << "Failed to parse " << ps.fileName << endl;
		return 1;
	}
	if(d.getInputSize() != ps.topology->front() || d.getOutputSize() != ps.topology->back()){
		cerr << "Topology incompatible with data." << endl;
		return 1;
	}

	Network nw (d.getInputSize(), *(ps.topology), ps.speed);
	nw.randomizeWeights(ps.minWeight, ps.maxWeight);
	for(int i = 0 ; i < ps.rounds; i++){
		nw.trainingEpoch(d);
	}
	epochStatus.inTraining = false;
	nw.testEpoch(d);
	return 0;
}

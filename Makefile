all: debug optimize

debug: bpann.h bpann.cpp Data.h Data.cpp Network.h Network.cpp
	g++ -g -std=c++11 -o NeuralNet bpann.cpp Data.cpp Network.cpp

optimize: bpann.h bpann.cpp Data.h Data.cpp Network.h Network.cpp
	g++ -std=c++11 -O3 -o NeuralNet_Opt bpann.cpp Data.cpp Network.cpp

clean:
	-@rm *.o NeuralNet NeuralNet_Opt 2>/dev/null

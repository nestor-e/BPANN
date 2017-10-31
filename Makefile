all: prgm

prgm: bpann.cpp
	g++ -std=c++11 -o BpNetwork bpann.cpp

clean:
	-@rm *.o BpNetwork

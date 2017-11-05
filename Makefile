all: prgm

prgm: bpann.h bpann.cpp Data.h Data.cpp Network.h Network.cpp
	g++ -g -std=c++11 -o BpNetwork bpann.cpp Data.cpp Network.cpp

clean:
	-@rm *.o BpNetwork 2>/dev/null

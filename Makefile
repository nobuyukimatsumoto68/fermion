CXX = g++-13
CXXFLAGS = -O3 -std=c++11 -fopenmp
INCLUDES = -I/usr/local/include/eigen-3.4.0

all: solve.o eps.o

solve.o: solve.cc header.hpp constants_and_typedefs.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

eps.o: eps_corr.cc header.hpp constants_and_typedefs.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

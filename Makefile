CXX = g++ # icpx # g++
CXXFLAGS = -O3 -std=c++17 -fopenmp # -openmp
INCLUDES = -I/projectnb/qfe/nmatsumo/opt/eigen/

NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70
INCLUDES_CUDA =


DIR = runfiles/


# all: solve.o solve.o eps.o tt.o
all: solve.o tt.o eps.o

solve.o: solve.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@

tt.o: tt_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

eps.o: eps_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

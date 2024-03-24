CXX = g++ # icpx # g++
CXXFLAGS = -O3 -std=c++17 -fopenmp # -openmp
INCLUDES = -I/projectnb/qfe/nmatsumo/opt/eigen/

NVCC = nvcc
NVCCFLAGS = -arch=sm_70 -O3 -lcusolver -std=c++17
INCLUDES_CUDA =


DIR = ./


# # all: solve.o solve.o eps.o tt.o

all: solve.o tt.o eps.o t_vev.o xixi.o eig.o
# all: tt.o eps.o t_vev.o psipsi.o eig.o

solve.o: solve.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@

tt.o: tt_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

t_vev.o: t_vev.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

xixi.o: xixi.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

eps.o: eps_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

# eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@


# all: eig.o eigen_matrix.o

eig.o: eig.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@

# eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

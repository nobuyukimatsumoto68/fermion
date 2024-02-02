CXX = g++
CXXFLAGS = -O3 -std=c++11 -fopenmp
INCLUDES = -I/usr/local/include/eigen-3.4.0

NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70
INCLUDES_CUDA =


# all: solve.o solve.o eps.o tt.o
all: solve.o solve_test.o tt.o

solve.o: solve.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $@

solve_test.o: solve_test.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

# solve_test.o: solve_test.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

tt.o: tt_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

# eps.o: eps_corr.cc header.hpp constants_and_typedefs.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $@

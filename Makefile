CXX = g++ # icpx # g++
CXXFLAGS = -O3 -std=c++17 -fopenmp # -openmp
CXXFLAGS0 = -O3 -std=c++17 # -openmp
INCLUDES = -I/opt/eigen/

NVCC = nvcc
NVCCFLAGS = -arch=sm_70 -O3 -lcusolver -std=c++17
INCLUDES_CUDA =
LDFLAGS_CUDA = -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib/

DIR = ./


# # all: solve.o solve.o eps.o tt.o

# all: solve.o eigen_matrix.o
all: solve.o tt.o eps.o t_vev.o t_vev_v2.o eps_vev.o xixi.o eig.o tt_v2.o t_epseps.o
# eigen_matrix.o
# all: solve.o acc.o cpu1.o cpuM.o
# all: tt.o eps.o t_vev.o psipsi.o eig.o

solve.o: solve.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp #
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@
cpu1.o: solve.cc header.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS0) $(INCLUDES) -o $(DIR)$@
cpuM.o: solve.cc header.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

tt.o: tt_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

tt_v2.o: tt_corr_v2.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

t_vev.o: t_vev.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

eps_vev.o: eps_vev.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

t_vev_v2.o: t_vev_v2.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

t_epseps.o: t_epseps.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

xixi.o: xixi.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

eps.o: eps_corr.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@


# all: eig.o eigen_matrix.o

eig.o: eig.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) $(LDFLAGS_CUDA) -o $(DIR)$@

# eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

acc.o: solve_acc2.cc header_acc.hpp constants.hpp
	nvc++ solve_acc2.cc -acc -O3 -I/opt/eigen/ -o $(DIR)$@

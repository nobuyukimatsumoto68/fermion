#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

// #include <omp.h>

// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

#include "typedefs_cuda.hpp"
#include "constants.hpp"
#include "header_cuda.hpp"

// ======================================



int main(){

  int device_num;
  cudacheck(cudaGetDeviceCount(&device_num));
  cudaDeviceProp device_prop[device_num];
  cudaGetDeviceProperties(&device_prop[0], 0);
  std::cout << "dev = " << device_prop[0].name << std::endl;
  cudacheck(cudaSetDevice(0));// "TITAN V"
  std::cout << "(GPU device is set.)" << std::endl;


  // -----

  const int xx = 0, yy = 0;

  Complex *e, *Dinv;
  e = (Complex*)malloc(N*CD);
  Dinv = (Complex*)malloc(N*CD);

  set2zero(e, N);
  e[ 2*idx(xx, yy) ] = cplx(1.0);
  multDdagger_wrapper( e, e );

  // set2zero(Dinv, N);
  // solve(Dinv, e);

  for(Idx i=0; i<N; i++){
    // std::cout << real(Dinv[i]) << " " << imag(Dinv[i]) << std::endl;
    std::cout << real(e[i]) << " " << imag(e[i]) << std::endl;
  }

  set2zero(e, N);
  e[ 2*idx(xx, yy)+1] = cplx(1.0);
  multDdagger_wrapper( e, e );

  // set2zero(Dinv, N);
  // solve(Dinv, e);

  for(Idx i=0; i<N; i++){
    // std::cout << real(Dinv[i]) << " " << imag(Dinv[i]) << std::endl;
    std::cout << real(e[i]) << " " << imag(e[i]) << std::endl;
  }

  free( e );
  free( Dinv );
  cudaDeviceReset();

  return 0;
}


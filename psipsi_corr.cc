#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <vector>

#include <omp.h>

#include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

#include "typedefs.hpp"
#include "constants.hpp"
#include "header.hpp"

// ======================================



int main(){

// #ifdef _OPENMP
//   omp_set_num_threads( 4 );
// #endif

  Vect Dinv0(2*Lx*Ly);
  Vect Dinv1(2*Lx*Ly);

  {
    std::ifstream ifs( dir_data+description+"Dinv0_cuda.dat",
                       std::ios::in | std::ios::binary );
    if(!ifs) assert(false);

    double real, imag;
    for(Idx i=0; i<2*Lx*Ly; ++i){
      ifs.read((char*) &real, sizeof(double) );
      ifs.read((char*) &imag, sizeof(double) );
      Dinv0[i] = real + I*imag;
    }
  }


  {
    std::ifstream ifs( dir_data+description+"Dinv1_cuda.dat",
                       std::ios::in | std::ios::binary );
    if(!ifs) assert(false);

    double real, imag;
    for(Idx i=0; i<2*Lx*Ly; ++i){
      ifs.read((char*) &real, sizeof(double) );
      ifs.read((char*) &imag, sizeof(double) );
      Dinv1[i] = real + I*imag;
    }
  }

  // -----------------------------------



  std::vector<M2> Dinv_n_0(Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      Dinv_n_0[idx(x,y)] <<
        Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
        Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
    }
  }

  M2 eps, eps_inv;
  eps << 0, 1, -1, 0;
  eps_inv << 0, -1, 1, 0;

  {
    std::ofstream of[4];
    for(int i=0; i<2; i++){
      for(int j=0; j<2; j++){
        const int ii = 2*i + j;
        of[ii].open( dir_data+description+"psipsi"+std::to_string(i)+std::to_string(j)+".dat",
                     std::ios::out | std::ios::trunc);
        if(!of[ii]) assert(false);
        of[ii] << std::scientific << std::setprecision(15);
      }}

    for(int y=0; y<Ly; y++){
      for(int x=0; x<Lx; x++){
        if(!is_site(x,y)) continue;
        const int c = mod(x-y, 3);
        M2 dinv = Dinv_n_0[idx(x,y)];

        for(int i=0; i<2; i++){
          for(int j=0; j<2; j++){
            const int ii = 2*i + j;
            const Complex tmp = dinv(i,j);
            of[ii] << x << " " << y << " " << tmp.real() << " " << tmp.imag() << " " << c << std::endl;
          }}
      }}
  }

  return 0;
}

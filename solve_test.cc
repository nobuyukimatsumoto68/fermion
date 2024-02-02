#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#include <omp.h>

#include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

#include "typedefs.hpp"
#include "constants.hpp"
#include "header.hpp"

// ======================================



int main(){

#ifdef _OPENMP
  omp_set_num_threads( nparallel );
#endif


  Vect init = Eigen::VectorXcd::Zero(2*Lx*Ly);

  const int xx = 0, yy = 0;

  Vect e0 = Eigen::VectorXcd::Zero(2*Lx*Ly);
  e0( 2*idx(xx, yy) ) = 1.0;
  e0 = multDdagger_eigen(e0);
  const Vect Dinv0 = CG(init, e0);

  std::cout << Dinv0 << std::endl;

  // {
  //   std::ofstream of( dir_data+description+"Dinv0.dat",
  //                     std::ios::out | std::ios::binary | std::ios::trunc);
  //   if(!of) assert(false);

  //   double tmp = 0.0;
  //   for(Idx i=0; i<2*Lx*Ly; i++){
  //     tmp = Dinv0(i).real();
  //     of.write((char*) &tmp, sizeof(double) );

  //     tmp = Dinv0(i).imag();
  //     of.write((char*) &tmp, sizeof(double) );
  //   }
  // }

  Vect e1 = Eigen::VectorXcd::Zero(2*Lx*Ly);
  e1( 2*idx(xx, yy)+1 ) = 1.0;
  e1 = multDdagger_eigen(e1);
  const Vect Dinv1 = CG(init, e1);

  std::cout << Dinv1 << std::endl;

  // {
  //   std::ofstream of( dir_data+description+"Dinv1.dat",
  //                     std::ios::out | std::ios::binary | std::ios::trunc);
  //   if(!of) assert(false);

  //   double tmp = 0.0;
  //   for(Idx i=0; i<2*Lx*Ly; i++){
  //     tmp = Dinv1(i).real();
  //     of.write((char*) &tmp, sizeof(double) );

  //     tmp = Dinv1(i).imag();
  //     of.write((char*) &tmp, sizeof(double) );
  //   }

  // }


  return 0;
}


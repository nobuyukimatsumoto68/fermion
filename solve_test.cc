#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#include <omp.h>

#include <Eigen/Dense>

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

  // e0 = multD_eigen(e0);
  const Vect Dinv0 = CG(init, e0);
  std::cout << Dinv0 << std::endl;
  // std::cout << e0 << std::endl;


  Vect e1 = Eigen::VectorXcd::Zero(2*Lx*Ly);
  e1( 2*idx(xx, yy)+1 ) = 1.0;
  e1 = multDdagger_eigen(e1);
  const Vect Dinv1 = CG(init, e1);
  std::cout << Dinv1 << std::endl;
  // std::cout << e1 << std::endl;

  return 0;
}


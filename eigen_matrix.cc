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

  Eigen::MatrixXcd matD1 = get_Dirac_matrix();
  Eigen::MatrixXcd matD( 2*Lx*Ly, 2*Lx*Ly );
  for(int i=0; i<2*Lx*Ly; i++){
    Vect e = Eigen::VectorXcd::Zero(2*Lx*Ly);
    e( i ) = 1.0;
    matD.block(0, i, 2*Lx*Ly, 1) = multD_eigen(e);
  }
  // std::cout << matD << std::endl;
  matD qw3

  return 0;
}


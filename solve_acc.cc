#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#include <Eigen/Dense>

#include "typedefs.hpp"
#include "constants.hpp"
#include "header_acc.hpp"

// ======================================



// int main(){
int main(int argc, char **argv){
#ifdef _OPENMP
  omp_set_num_threads( nparallel );
#endif

  if (argc>1){
    nu = atoi(argv[1]);
    // printf("%s\n", argv[i]);
  }

  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);

  Vect init = Eigen::VectorXcd::Zero(2*Lx*Ly);

  const int xx = 0, yy = 0;

  Vect e0 = Eigen::VectorXcd::Zero(2*Lx*Ly);
  e0( 2*idx(xx, yy) ) = 1.0;
  e0 = multDdagger_eigen(e0);

  // std::cout << "e0 = " << e0.transpose() << std::endl;

  const Vect Dinv0 = CG(init, e0);

  // std::cout << "Dinv0 = " << Dinv0 << std::endl;
  // std::cout << "D Dinv0 = " << multD_eigen(Dinv0) << std::endl;

  {
    std::ofstream of( dir_data+description+"Dinv0.dat",
                      std::ios::out | std::ios::binary | std::ios::trunc);
    if(!of) assert(false);

    double tmp = 0.0;
    for(Idx i=0; i<2*Lx*Ly; i++){
      tmp = Dinv0(i).real();
      of.write((char*) &tmp, sizeof(double) );

      tmp = Dinv0(i).imag();
      of.write((char*) &tmp, sizeof(double) );
    }
  }

  Vect e1 = Eigen::VectorXcd::Zero(2*Lx*Ly);
  e1( 2*idx(xx, yy)+1 ) = 1.0;
  e1 = multDdagger_eigen(e1);
  const Vect Dinv1 = CG(init, e1);

  {
    std::ofstream of( dir_data+description+"Dinv1.dat",
                      std::ios::out | std::ios::binary | std::ios::trunc);
    if(!of) assert(false);

    double tmp = 0.0;
    for(Idx i=0; i<2*Lx*Ly; i++){
      tmp = Dinv1(i).real();
      of.write((char*) &tmp, sizeof(double) );

      tmp = Dinv1(i).imag();
      of.write((char*) &tmp, sizeof(double) );
    }

  }


  return 0;
}


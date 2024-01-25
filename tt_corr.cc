#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <vector>

#include <omp.h>

#include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

#include "constants_and_typedefs.hpp"
#include "header.hpp"

// ======================================



int main(){

// #ifdef _OPENMP
//   omp_set_num_threads( 4 );
// #endif

  Vect Dinv0(2*Lx*Ly);
  Vect Dinv1(2*Lx*Ly);

  {
    std::ifstream ifs( dir_data+description+"Dinv0.dat",
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
    std::ifstream ifs( dir_data+description+"Dinv1.dat",
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

  // PACKAGE IT!!!

  M2 eps, eps_inv;
  eps << 0, 1, -1, 0;
  eps_inv << 0, -1, 1, 0;

  std::array<M2, 6> gamma;
  {
    for(int mu=0; mu<SIX; mu++){
      V2 emu = get_e( mu );
      gamma[mu] = e(0)*sigma[2] + e(1)*sigma[1];
    }
  }

  // PACKAGE IT!!!

  {
    std::ofstream of( dir_data+description+"tt.dat",
                      std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    for(int y=0; y<Ly; y++){
      for(int x=0; x<Lx; x++){
        if(!is_site(x,y)) continue;
        const int c = mod(x-y, 3);
        if(c==0){
          M2 dinv = Dinv_n_0[idx(x,y)];

          for(int sigma=0; sigma<THREE; sigma++){
            for(int nu=0; nu<THREE; nu++){
              int xp, yp;
              cshift(sp, yp, x, y, sigma);
              cshift_minus(xp, yp, xp, yp, nu);
              M2 dinv_p = Dinv_n_0[idx(xp,yp)];

              for(int mu=0; mu<THREE; mu++){
                for(int rho=0; rho<THREE; rho++){
                  M2 tmp = eps_inv * gamma[mu].transpose() * dinv.transpose() * gamma[rho].transpose() * eps * dinv;
                  Complex corr = 2.0 * tmp.trace();
                  of << x << " " << y << " "
                     << mu << " " << nu << " " << rho << " " << sigma << " "
                     << corr.real() << " " << corr.imag() << " " << c << std::endl;
                }}
            }}
        }
      }}
  }


  return 0;
}

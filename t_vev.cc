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

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads( nparallel );
#endif

  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);

  std::vector<M2> Dinv_n_0(Lx*Ly), Dinv_n_A(Lx*Ly), Dinv_n_B(Lx*Ly), Dinv_n_C(Lx*Ly);

  {
    Vect Dinv0(2*Lx*Ly);
    Vect Dinv1(2*Lx*Ly);

    {
      std::ifstream ifs( dir_data+description+"Dinv_0_0_0_cuda.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_0_0_1_cuda.dat",
                         // std::ifstream ifs( dir_data+description+"Dinv1_cuda.dat",
                         std::ios::in | std::ios::binary );
      if(!ifs) assert(false);

      double real, imag;
      for(Idx i=0; i<2*Lx*Ly; ++i){
        ifs.read((char*) &real, sizeof(double) );
        ifs.read((char*) &imag, sizeof(double) );
        Dinv1[i] = real + I*imag;
      }
    }


    for(int x=0; x<Lx; x++){
      for(int y=0; y<Ly; y++){
        Dinv_n_0[idx(x,y)] <<
          Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
          Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
      }
    }
  }


  // -----------------------------------

  {
    Vect Dinv0(2*Lx*Ly);
    Vect Dinv1(2*Lx*Ly);

    {
      std::ifstream ifs( dir_data+description+"Dinv_m1_0_0_cuda.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_m1_0_1_cuda.dat",
                         std::ios::in | std::ios::binary );
      if(!ifs) assert(false);

      double real, imag;
      for(Idx i=0; i<2*Lx*Ly; ++i){
        ifs.read((char*) &real, sizeof(double) );
        ifs.read((char*) &imag, sizeof(double) );
        Dinv1[i] = real + I*imag;
      }
    }


    for(int x=0; x<Lx; x++){
      for(int y=0; y<Ly; y++){
        Dinv_n_A[idx(x,y)] <<
          Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
          Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
      }
    }
  }


  // -----------------------------------

  {
    Vect Dinv0(2*Lx*Ly);
    Vect Dinv1(2*Lx*Ly);

    {
      std::ifstream ifs( dir_data+description+"Dinv_1_m1_0_cuda.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_1_m1_1_cuda.dat",
                         std::ios::in | std::ios::binary );
      if(!ifs) assert(false);

      double real, imag;
      for(Idx i=0; i<2*Lx*Ly; ++i){
        ifs.read((char*) &real, sizeof(double) );
        ifs.read((char*) &imag, sizeof(double) );
        Dinv1[i] = real + I*imag;
      }
    }


    for(int x=0; x<Lx; x++){
      for(int y=0; y<Ly; y++){
        Dinv_n_B[idx(x,y)] <<
          Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
          Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
      }
    }
  }


  // -----------------------------------

  {
    Vect Dinv0(2*Lx*Ly);
    Vect Dinv1(2*Lx*Ly);

    {
      std::ifstream ifs( dir_data+description+"Dinv_0_1_0_cuda.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_0_1_1_cuda.dat",
                         std::ios::in | std::ios::binary );
      if(!ifs) assert(false);

      double real, imag;
      for(Idx i=0; i<2*Lx*Ly; ++i){
        ifs.read((char*) &real, sizeof(double) );
        ifs.read((char*) &imag, sizeof(double) );
        Dinv1[i] = real + I*imag;
      }
    }


    for(int x=0; x<Lx; x++){
      for(int y=0; y<Ly; y++){
        Dinv_n_C[idx(x,y)] <<
          Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
          Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
      }
    }
  }


  // -----------------------------------

  M2 eps = get_eps();
  M2 eps_inv = -eps;

  std::array<M2, 6> gamma;
  for(int b=0; b<SIX; b++) gamma[b] = get_gamma(b);

  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int m=0; m<THREE; m++){
      for(int n=0; n<THREE; n++){

        std::ofstream of( dir_data+description+"t_vev"+std::to_string(m)+std::to_string(n)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        int x=0, y=0;
        if(!is_site(x,y)) assert(false);

        const int ch = mod(x-y, 3);
        if(ch!=0) assert(false);

        int xpn, ypn;
        int sign = cshift(xpn, ypn, x, y, n);

        std::vector<M2> Dinv_n_0pn;
        if(n==0) Dinv_n_0pn = Dinv_n_A;
        else if(n==1) Dinv_n_0pn = Dinv_n_B;
        else if(n==2) Dinv_n_0pn = Dinv_n_C;
        else assert(false);

        const M2& Dinv_0_0pn = sign * Dinv_n_0pn[idx(x,y)];

        const Complex tr = -( gamma[m] * Dinv_0_0pn ).trace();
        of << tr.real() << " " << tr.imag() << " "
           << m << " " << n << " " << std::endl;
      }}
  }

  return 0;
}

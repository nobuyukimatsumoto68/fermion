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

  // PACKAGE IT!!!

  M2 eps, eps_inv;
  eps << 0, 1, -1, 0;
  eps_inv << 0, -1, 1, 0;

  std::array<M2, 6> gamma;
  {
    for(int b=0; b<SIX; b++){
      V2 e = get_e( b );
      gamma[b] = e(0)*sigma[2] + e(1)*sigma[1];
    }
  }

//   {
//     const M2 eps_gamma_a = eps_inv.transpose() * sigma[2].transpose();
//     const M2 eps_gamma_c = eps * sigma[2];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//     for(int s=0; s<THREE; s++){
//       for(int b=0; b<THREE; b++){
//         {
//           // inside loop
//           std::ofstream of( dir_data+description+"t0"+std::to_string(s)+".dat",
//                             std::ios::out | std::ios::trunc);
//           if(!of) assert(false);
//           of << std::scientific << std::setprecision(15);

//           // inside loop
//           for(int y=0; y<Ly; y++){
//             for(int x=0; x<Lx; x++){
//               if(!is_site(x,y)) continue;

//               const int ch = mod(x-y, 3);
//               if(ch==0){
//                 const M2 dinv_x = Dinv_n_0[idx(x,y)];

//                 int xps, yps;
//                 int sign1 = cshift(xps, yps, x, y, s);
//                 const M2 dinv_xps = sign1 * Dinv_n_0[idx(xps,yps)];

//                 int xmb, ymb;
//                 int sign2 = cshift_minus(xmb, ymb, x, y, b);
//                 int xpsmb, ypsmb;
//                 int sign3 = cshift(xpsmb, ypsmb, xmb, ymb, s);
//                 const M2 dinv_xmb = sign2*Dinv_n_0[idx(xmb,ymb)];
//                 const M2 dinv_xpsmb = sign2*sign3*Dinv_n_0[idx(xpsmb,ypsmb)];

//                 const Complex tr1 = ( eps_gamma_a * dinv_x.transpose() * eps_gamma_c * dinv_xpsmb ).trace();
//                 const Complex tr2 = ( eps_gamma_a * dinv_xps.transpose() * eps_gamma_c * dinv_xmb ).trace();
//                 const Complex corr = tr1 - tr2;
//                 of << x << " " << y << " "
//                    << corr.real() << " " << corr.imag() << " "
//                    << ch << " " << s << " " << b << " " << std::endl;
//               } // end if
//             }} // end for x,y
//         }
//       }}
//   }

  // PACKAGE IT!!!

  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int a=0; a<THREE; a++){
      for(int b=0; b<THREE; b++){

        std::ofstream of( dir_data+description+"t_vev"+std::to_string(a)+std::to_string(b)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        int x=0, y=0;
        if(!is_site(x,y)) assert(false);

        const int ch = mod(x-y, 3);
        if(ch!=0) assert(false);

        int xpb, ypb;
        int sign1 = cshift(xpb, ypb, x, y, b);
        const M2 dinv_xpb = sign1 * Dinv_n_0[idx(xpb,ypb)];

        const Complex tr = ( gamma[a] * dinv_xpb ).trace();
        of << tr.real() << " " << tr.imag() << " "
           << ch << " " << a << " " << b << " " << std::endl;
      }}
  }


  return 0;
}
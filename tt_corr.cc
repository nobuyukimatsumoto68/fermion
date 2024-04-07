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



  // PACKAGE IT!!!
  // M2 eps, eps_inv;
  // eps << 0, 1, -1, 0;
  // eps_inv << 0, -1, 1, 0;
  M2 eps = get_eps();
  M2 eps_inv = -eps;

  std::array<M2, 6> gamma;
  {
    for(int b=0; b<SIX; b++){
      // V2 e = get_e( b );
      gamma[b] = get_gamma(b);
    }
  }


  // std::vector<M2> Dinv_n_0(Lx*Ly), Dinv_n_A(Lx*Ly), Dinv_n_B(Lx*Ly), Dinv_n_C(Lx*Ly)
  // 00 - 00
  {
    // const int m=0, n=0, r=0, s=0;
#ifdef _OPENMP
#pragma omp parallel for collapse(4) num_threads(nparallel)
#endif
    for(int m=0; m<THREE; m++){
      for(int s=0; s<THREE; s++){
        for(int r=0; r<THREE; r++){
          for(int n=0; n<THREE; n++){

            int dum1,dum2;
            int sign1 = cshift(dum1, dum2, 0, 0, n);

            std::vector<M2> Dinv_n_0pn;
            if(n==0) Dinv_n_0pn = Dinv_n_A;
            else if(n==1) Dinv_n_0pn = Dinv_n_B;
            else if(n==2) Dinv_n_0pn = Dinv_n_C;
            else assert(false);

            std::ofstream of( dir_data+description+"tt"+std::to_string(m)+std::to_string(n)+std::to_string(r)+std::to_string(s)+".dat",
                              std::ios::out | std::ios::trunc);
            if(!of) assert(false);
            of << std::scientific << std::setprecision(15);

            for(int y=0; y<Ly; y++){
              for(int x=0; x<Lx; x++){
                if(!is_site(x,y)) continue;
                const int ch = mod(x-y, 3);
                if(ch!=0) continue;

                const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
                const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

                int xps, yps;
                int sign2 = cshift(xps, yps, x, y, s);
                const M2& Dinv_xps_0 = sign2 * Dinv_n_0[idx(xps,yps)];
                const M2& Dinv_xps_0pn = sign1 * sign2 * Dinv_n_0pn[idx(xps,yps)];

                const Complex tr1 = ( Dinv_x_0pn * gamma[m] * eps_inv * Dinv_xps_0.transpose() * eps * gamma[r] ).trace();
                const Complex tr2 = ( Dinv_xps_0pn * gamma[m] * eps_inv * Dinv_x_0.transpose() * eps * gamma[r] ).trace();
                const Complex corr = - tr1 + tr2;

                of << x << " " << y << " "
                   << m << " " << n << " " << r << " " << s << " "
                   << corr.real() << " " << corr.imag() << " "
                   << std::endl;
              }}
          }}}}
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
//           std::ofstream of( dir_data+description+"t0"+std::to_string(s)+"t0"+std::to_string(b)+".dat",
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

//   {
// #ifdef _OPENMP
// #pragma omp parallel for collapse(4) num_threads(nparallel)
// #endif
//     for(int a=0; a<THREE; a++){
//       for(int c=0; c<THREE; c++){
//         for(int s=0; s<THREE; s++){
//           for(int b=0; b<THREE; b++){

//             const M2 eps_gamma_a = eps_inv.transpose() * gamma[a].transpose();
//             const M2 eps_gamma_c = eps * gamma[c];

// #ifdef _OPENMP
// #pragma omp critical
// #endif
//             { std::clog << "a = " << a << ", b = " << b << ", c = " << c << ", d = " << s << std::endl; }

//             // inside loop
//             std::ofstream of( dir_data+description+"tt"+std::to_string(a)+std::to_string(b)+std::to_string(c)+std::to_string(s)+".dat",
//                               std::ios::out | std::ios::trunc);
//             if(!of) assert(false);
//             of << std::scientific << std::setprecision(15);
//             // Complex tmp[Lx][Ly];

//             // inside loop
//             for(int y=0; y<Ly; y++){
//               for(int x=0; x<Lx; x++){
//                 if(!is_site(x,y)) continue;

//                 const int ch = mod(x-y, 3);
//                 if(ch==0){
//                   const M2 dinv_x = Dinv_n_0[idx(x,y)];

//                   int xps, yps;
//                   int sign1 = cshift(xps, yps, x, y, s);
//                   const M2 dinv_xps = sign1 * Dinv_n_0[idx(xps,yps)];

//                   int xmb, ymb;
//                   int sign2 = cshift_minus(xmb, ymb, x, y, b);
//                   int xpsmb, ypsmb;
//                   int sign3 = cshift(xpsmb, ypsmb, xmb, ymb, s);
//                   const M2 dinv_xmb = sign2*Dinv_n_0[idx(xmb,ymb)];
//                   const M2 dinv_xpsmb = sign2*sign3*Dinv_n_0[idx(xpsmb,ypsmb)];

//                   const Complex tr1 = ( eps_gamma_a * dinv_x.transpose() * eps_gamma_c * dinv_xpsmb ).trace();
//                   const Complex tr2 = ( eps_gamma_a * dinv_xps.transpose() * eps_gamma_c * dinv_xmb ).trace();
//                   const Complex corr = tr1 - tr2;
//                   // tmp[x][y][a][b][c][s] = tmp1.trace() - tmp2.trace();
//                   of << x << " " << y << " "
//                      << corr.real() << " " << corr.imag() << " "
//                      << ch << " " << a << " " << b << " " << c << " " << s << " " << std::endl;
//                 } // end if

//               }} // end for x,y
//           }}
//       }}
//   }


  return 0;
}

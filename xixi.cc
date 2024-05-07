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



//int main(){
int main(int argc, char **argv){

// #ifdef _OPENMP
//   omp_set_num_threads( 4 );
// #endif
  if (argc>1){
    nu = atoi(argv[1]);
    // printf("%s\n", argv[i]);
  }
  // description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);

  set_all();


  {
    Vect Dinv0(2*Lx*Ly);
    Vect Dinv1(2*Lx*Ly);

    {
      std::ifstream ifs( dir_data+description+"Dinv_0_0_0.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_0_0_1.dat",
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

    {
      std::ofstream of[4];
      for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
          const int ii = 2*i + j;
          of[ii].open( dir_data+description+"xixi"+std::to_string(i)+std::to_string(j)+".dat",
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

  }


  return 0;
}





// //----------------------------


// {

//   Vect Dinv0(2*Lx*Ly);
//   Vect Dinv1(2*Lx*Ly);

//   {
//     std::ifstream ifs( dir_data+description+"Dinv_m1_0_0.dat",
//                        std::ios::in | std::ios::binary );
//     if(!ifs) assert(false);

//     double real, imag;
//     for(Idx i=0; i<2*Lx*Ly; ++i){
//       ifs.read((char*) &real, sizeof(double) );
//       ifs.read((char*) &imag, sizeof(double) );
//       Dinv0[i] = real + I*imag;
//     }
//   }


//   {
//     std::ifstream ifs( dir_data+description+"Dinv_m1_0_1.dat",
//                        std::ios::in | std::ios::binary );
//     if(!ifs) assert(false);

//     double real, imag;
//     for(Idx i=0; i<2*Lx*Ly; ++i){
//       ifs.read((char*) &real, sizeof(double) );
//       ifs.read((char*) &imag, sizeof(double) );
//       Dinv1[i] = real + I*imag;
//     }
//   }

//   // -----------------------------------



//   std::vector<M2> Dinv_n_0(Lx*Ly);

//   for(int x=0; x<Lx; x++){
//     for(int y=0; y<Ly; y++){
//       Dinv_n_0[idx(x,y)] <<
//         Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
//         Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
//     }
//   }

//   {
//     std::ofstream of[4];
//     for(int i=0; i<2; i++){
//       for(int j=0; j<2; j++){
//         const int ii = 2*i + j;
//         of[ii].open( dir_data+description+"xixi_m1_0_"+std::to_string(i)+std::to_string(j)+".dat",
//                      std::ios::out | std::ios::trunc);
//         if(!of[ii]) assert(false);
//         of[ii] << std::scientific << std::setprecision(15);
//       }}

//     for(int y=0; y<Ly; y++){
//       for(int x=0; x<Lx; x++){
//         if(!is_site(x,y)) continue;
//         const int c = mod(x-y, 3);
//         M2 dinv = Dinv_n_0[idx(x,y)];

//         for(int i=0; i<2; i++){
//           for(int j=0; j<2; j++){
//             const int ii = 2*i + j;
//             const Complex tmp = dinv(i,j);
//             of[ii] << x << " " << y << " " << tmp.real() << " " << tmp.imag() << " " << c << std::endl;
//           }}
//       }}
//   }

// }

// {

//   Vect Dinv0(2*Lx*Ly);
//   Vect Dinv1(2*Lx*Ly);

//   {
//     std::ifstream ifs( dir_data+description+"Dinv_0_1_0.dat",
//                        std::ios::in | std::ios::binary );
//     if(!ifs) assert(false);

//     double real, imag;
//     for(Idx i=0; i<2*Lx*Ly; ++i){
//       ifs.read((char*) &real, sizeof(double) );
//       ifs.read((char*) &imag, sizeof(double) );
//       Dinv0[i] = real + I*imag;
//     }
//   }


//   {
//     std::ifstream ifs( dir_data+description+"Dinv_0_1_1.dat",
//                        std::ios::in | std::ios::binary );
//     if(!ifs) assert(false);

//     double real, imag;
//     for(Idx i=0; i<2*Lx*Ly; ++i){
//       ifs.read((char*) &real, sizeof(double) );
//       ifs.read((char*) &imag, sizeof(double) );
//       Dinv1[i] = real + I*imag;
//     }
//   }

//   // -----------------------------------



//   std::vector<M2> Dinv_n_0(Lx*Ly);

//   for(int x=0; x<Lx; x++){
//     for(int y=0; y<Ly; y++){
//       Dinv_n_0[idx(x,y)] <<
//         Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
//         Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
//     }
//   }

//   {
//     std::ofstream of[4];
//     for(int i=0; i<2; i++){
//       for(int j=0; j<2; j++){
//         const int ii = 2*i + j;
//         of[ii].open( dir_data+description+"xixi_0_1_"+std::to_string(i)+std::to_string(j)+".dat",
//                      std::ios::out | std::ios::trunc);
//         if(!of[ii]) assert(false);
//         of[ii] << std::scientific << std::setprecision(15);
//       }}

//     for(int y=0; y<Ly; y++){
//       for(int x=0; x<Lx; x++){
//         if(!is_site(x,y)) continue;
//         const int c = mod(x-y, 3);
//         M2 dinv = Dinv_n_0[idx(x,y)];

//         for(int i=0; i<2; i++){
//           for(int j=0; j<2; j++){
//             const int ii = 2*i + j;
//             const Complex tmp = dinv(i,j);
//             of[ii] << x << " " << y << " " << tmp.real() << " " << tmp.imag() << " " << c << std::endl;
//           }}
//       }}
//   }

// }

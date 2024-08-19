#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

// #include <omp.h>

// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

#include "typedefs_cuda.hpp"
#include "constants.hpp"
#include "header_cuda.hpp"

// ======================================



// int main(){
int main(int argc, char **argv){

  if (argc>1){
    nu = atoi(argv[1]);
    // printf("%s\n", argv[i]);
  }
  // if (argc==3){
  //   Lx = atoi(argv[2]);
  //   Ly = Lx;
  // }
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"theta"+std::to_string(theta);
  set_all();
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2);
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2)+str(is_periodic_orthogonal);
  // description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);

  // Idx xP = Lx/2, yP = 0;

  int device_num;
  cudacheck(cudaGetDeviceCount(&device_num));
  cudaDeviceProp device_prop[device_num];
  cudaGetDeviceProperties(&device_prop[0], 0);
  std::cout << "dev = " << device_prop[0].name << std::endl;
  cudacheck(cudaSetDevice(0));// "TITAN V"
  std::cout << "(GPU device is set.)" << std::endl;


  std::cout << "ell0 = " << ell0[0] << ", " << ell0[1] << std::endl
            << "ell1 = " << ell1[0] << ", " << ell1[1] << std::endl
            << "ell2 = " << ell2[0] << ", " << ell2[1] << std::endl;

  std::cout << "ell = " << ell[0] << ", " << ell[1] << ", " << ell[2] << std::endl;

  std::cout << "kappa = " << kappa[0] << ", " << kappa[1] << ", " << kappa[2] << std::endl;

  std::cout << "ell0* = " << ell_star0[0] << ", " << ell_star0[1] << std::endl
            << "ell1* = " << ell_star1[0] << ", " << ell_star1[1] << std::endl
            << "ell2* = " << ell_star2[0] << ", " << ell_star2[1] << std::endl;

  std::cout << "e0 = " << e0[0] << ", " << e0[1] << std::endl
            << "e1 = " << e1[0] << ", " << e1[1] << std::endl
            << "e2 = " << e2[0] << ", " << e2[1] << std::endl;

  {
    std::ofstream of;
    of.open( dir_data+description+"ell.dat", std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    of << ell0[0] << ", " << ell0[1] << std::endl
       << ell1[0] << ", " << ell1[1] << std::endl
       << ell2[0] << ", " << ell2[1] << std::endl;
  }

  {
    std::ofstream of;
    of.open( dir_data+description+"kappa.dat", std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    of << kappa[0] << ", " << kappa[1] << ", " << kappa[2] << std::endl;
  }

  {
    std::ofstream of;
    of.open( dir_data+description+"ellstar.dat", std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    of << ell_star0[0] << ", " << ell_star0[1] << std::endl
       << ell_star1[0] << ", " << ell_star1[1] << std::endl
       << ell_star2[0] << ", " << ell_star2[1] << std::endl;
  }

  {
    std::ofstream of;
    of.open( dir_data+description+"e.dat", std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    // of << ell_star0[0] << ", " << ell_star0[1] << std::endl
    //    << ell_star1[0] << ", " << ell_star1[1] << std::endl
    //    << ell_star2[0] << ", " << ell_star2[1] << std::endl;
    of << e0[0] << ", " << e0[1] << std::endl
       << e1[0] << ", " << e1[1] << std::endl
       << e2[0] << ", " << e2[1] << std::endl;

  }


  // return 1;


  // -----

  Complex *e, *Dinv;
  e = (Complex*)malloc(N*CD);
  Dinv = (Complex*)malloc(N*CD);

  {
    int xx = 0, yy = 0;

    set2zero(e, N);
    e[ 2*idx(xx, yy) ] = cplx(1.0);
    multDdagger_wrapper( e, e);

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_0_0_0.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }

    set2zero(e, N);
    e[ 2*idx(xx, yy)+1] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_0_0_1.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }
  }

  //------------------------------------------

  {
    // int xx = -1, yy = 0;
    int xx = xs, yy = ys;
    cshift( xx, yy, xx, yy, 0, nu );

    set2zero(e, N);
    e[ 2*idx(xx, yy) ] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_m1_0_0.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }

    set2zero(e, N);
    e[ 2*idx(xx, yy)+1] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_m1_0_1.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }
  }

  //------------------------------------------

  {
    // int xx = 1, yy = -1;
    int xx = xs, yy = ys;
    cshift( xx, yy, xx, yy, 1, nu );

    set2zero(e, N);
    e[ 2*idx(xx, yy) ] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_1_m1_0.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }

    set2zero(e, N);
    e[ 2*idx(xx, yy)+1] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_1_m1_1.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }
  }

  //------------------------------------------

  {
    // int xx = 0, yy = 1;
    int xx = xs, yy = ys;
    cshift( xx, yy, xx, yy, 2, nu );

    set2zero(e, N);
    e[ 2*idx(xx, yy) ] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_0_1_0.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }

    set2zero(e, N);
    e[ 2*idx(xx, yy)+1] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_0_1_1.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }
  }

  //

  {
    int xx = xP, yy = yP;

    set2zero(e, N);
    e[ 2*idx(xx, yy) ] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_xP_yP_0.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }

    set2zero(e, N);
    e[ 2*idx(xx, yy)+1] = cplx(1.0);
    multDdagger_wrapper( e, e );

    set2zero(Dinv, N);
    solve(Dinv, e);

    {
      std::ofstream of( dir_data+description+"Dinv_xP_yP_1.dat",
                        std::ios::out | std::ios::binary | std::ios::trunc);
      if(!of) assert(false);

      double tmp = 0.0;
      for(Idx i=0; i<N; i++){
        tmp = real(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );

        tmp = imag(Dinv[i]);
        of.write((char*) &tmp, sizeof(double) );
      }
    }
  }

  free( e );
  free( Dinv );




  cudaDeviceReset();

  return 0;
}


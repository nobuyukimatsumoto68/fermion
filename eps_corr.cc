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



// int main(){
int main(int argc, char **argv){

// #ifdef _OPENMP
//   omp_set_num_threads( 4 );
// #endif

  Vect Dinv0(2*Lx*Ly);
  Vect Dinv1(2*Lx*Ly);

  if (argc>1){
    nu = atoi(argv[1]);
    // printf("%s\n", argv[i]);
  }
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"theta"+std::to_string(theta);
  set_all();
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2);
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2)+str(is_periodic_orthogonal);

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


  // const int xx = 0, yy = 0;
  // std::cout << "(x,y) = (" << xx << ", " << yy << "):" << std::endl
  //           << "Dinv0 = " << std::endl
  //           << Dinv0( 2*idx(xx,yy) ) << ", "
  //           << Dinv0( 2*idx(xx,yy)+1 )
  //           << std::endl;

  std::vector<M2> Dinv_n_0(Lx*Ly);

  // int xx = 0, yy = 0;
  // std::cout << "# Lx, Ly = " << Lx << ", Ly = " << Ly << std::endl;

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      Dinv_n_0[idx(x,y)] <<
        Dinv0( 2*idx(x,y) ), Dinv1( 2*idx(x,y) ),
        Dinv0( 2*idx(x,y)+1 ), Dinv1( 2*idx(x,y)+1 );
        // << Dinv0( 2*idx(x,y)+1 ).real() + I*Dinv0( 2*idx(x,y)+1 ).imag()
        // << Dinv1( 2*idx(x,y)+1 ).real() + I*Dinv1( 2*idx(x,y)+1 ).imag();

      // if(!is_site(xx,yy)) continue;
      // std::cout << xx
      //           << " "
      //           << Dinv0( 2*idx(xx,yy) ).real()
      //           << " "
      //           << Dinv0( 2*idx(xx,yy) ).imag()
      //           << " "
      //           << Dinv0( 2*idx(xx,yy)+1 ).real()
      //           << " "
      //           << Dinv0( 2*idx(xx,yy)+1 ).imag()
      //           << std::endl;
    }
  }

  // M2 eps, eps_inv;
  // eps << 0, 1, -1, 0;
  // eps_inv << 0, -1, 1, 0;
  M2 eps = get_eps();
  M2 eps_inv = -eps;

  {
    std::ofstream of( dir_data+description+"eps_corr.dat",
                      std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    for(int y=0; y<Ly; y++){
      for(int x=0; x<Lx; x++){
        if(!is_site(x,y)) continue;
        const int c = mod(x-y, 3);
        M2 dinv = Dinv_n_0[idx(x,y)];
        M2 tmp = eps_inv.transpose() * dinv.transpose() * eps * dinv;
        Complex corr = 0.5 * tmp.trace();
        of << x << " " << y << " " << corr.real() << " " << corr.imag() << " " << c << std::endl;
      }}
  }

  // for(int x=0; x<Lx; x++){
  //   for(int y=0; y<Ly; y++){
  //     if(!is_site(x,y)) continue;
  //     M2 dinv = Dinv_n_0[idx(x,y)];
  //     M2 tmp = eps_inv.transpose() * dinv.transpose() * eps * dinv;
  //     Complex corr = 2.0 * tmp.trace();
  //     std::cout << x << " " << y << " " << corr.real() << " " << corr.imag() << std::endl;
  //   }}


  // std::cout << "Dinv1 = " << std::endl
  //           << Dinv1( 2*idx(xx,yy) ) << ", "
  //           << Dinv1( 2*idx(xx,yy)+1 )
  //           << std::endl;

  return 0;
}

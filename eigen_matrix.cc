#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "typedefs.hpp"
#include "constants.hpp"
#include "header.hpp"

// ======================================



int main(){

#ifdef _OPENMP
  omp_set_num_threads( nparallel );
#endif

  std::cout << std::scientific << std::setprecision(15) << std::endl;

  Eigen::MatrixXcd matD0 = get_Dirac_matrix();

  Eigen::MatrixXcd matD( 2*Lx*Ly, 2*Lx*Ly );
  for(int i=0; i<2*Lx*Ly; i++){
    Vect e = Eigen::VectorXcd::Zero(2*Lx*Ly);
    e( i ) = 1.0;
    matD.block(0, i, 2*Lx*Ly, 1) = multD_eigen(e);
  }
  Eigen::MatrixXcd diff = matD - matD0;
  std::cout << "diff = " << (diff*diff.adjoint()).trace() << std::endl;

  Eigen::MatrixXcd eps = get_large_epsilon();
  Eigen::MatrixXcd A = eps*matD;
  diff = A + A.transpose();
  std::cout << "diff = " << (diff*diff.adjoint()).trace() << std::endl;

  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
  ces.compute( A );
  Eigen::VectorXcd ev = ces.eigenvalues();
  // std::cout << "The eigenvalues of A are:" << std:: endl
  //           << ev << std::endl;
  // std::cout << matD << std::endl;

  int i0=0, i=0;
  std::complex det = 1.0;
  for(auto elem : ev ){
    if( std::abs(elem)<1.0e-14 ){
      i0++;
    }
    else{
      i++;
      // std::cout << "elem = " << elem << std::endl;
      det *= elem;
      // std::cout << "det = " << det << std::endl;
    }
  }
  std::cout << "i0 = " << i0 << std::endl
            << "i = " << i << std::endl
            << "det = " << det << std::endl
            << "Pf = " << std::sqrt( det.real() ) << std::endl;
            // << "Pf/3 = " << std::pow(2,24) * std::sqrt( det.real() ) / 3.0 << std::endl;

  Eigen::MatrixXcd P,Q,R,PI,QI,RI;
  P = Wilson_projector(0);
  Q = Wilson_projector(1);
  R = Wilson_projector(2);
  PI = Wilson_projector(3);
  QI = Wilson_projector(4);
  RI = Wilson_projector(5);

  // std::cout << "2P = " << 2*P << std::endl;
  // std::cout << "2Q = " << 2*Q << std::endl;
  // std::cout << "2R = " << 2*R << std::endl;
  // std::cout << "2PI = " << 2*PI << std::endl;
  // std::cout << "2QI = " << 2*QI << std::endl;
  // std::cout << "2RI = " << 2*RI << std::endl;

  {
    // double Z = 1.0;
    Eigen::MatrixXcd C;
    std::complex<double> tr, diff, Z;
    double check;
    int len;
    Z = 1.0;

    // {
    //   C = R*PI*Q*RI*P*QI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   diff = std::pow(kappa,len) * tr - check;
    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += check;
    // }

    // {
    //   C = R*QI*P*RI*Q*PI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   diff = std::pow(kappa,len) * tr - check;
    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += check;
    // }

    // {
    //   C = P*RI*Q*PI*R*QI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   diff = std::pow(kappa,len) * tr - check;
    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += check;
    // }

    {
      // C = -R*PI*R*QI;
      C = R*PI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      // C = -QI*R*PI*R;
      C = QI*R*PI*R;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      // C = -R*QI*R*PI;
      C = R*QI*R*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      // C = -PI*R*PI*Q;
      C = PI*R*PI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      // C = -P*RI*P*QI;
      C = P*RI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      // C = -R*PI*Q*PI;
      C = R*PI*Q*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      C = R*QI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      C = P*QI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      C = PI*Q*RI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    // {
    //   C = -R*QI*R*QI*R*QI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   diff = std::pow(kappa,len) * tr - check;
    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += check;
    // }

    // {
    //   C = R*PI*R*PI*R*PI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   int special_sign = -1.0;

    //   diff = std::pow(kappa,len) * tr - special_sign*check;
    //   // std::cout << "mtr = " << std::pow(kappa,len) * tr << std::endl
    //   //           << "ch = " << special_sign*check << std::endl;

    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += special_sign*check;
    // }

    // {
    //   C = -P*QI*P*QI*P*QI;
    //   len = 6;
    //   tr = -C.trace();
    //   check = std::pow( kappa*std::cos(M_PI/6.0), len);

    //   diff = std::pow(kappa,len) * tr - check;
    //   assert( std::abs(diff)<1.0e-14 );
    //   Z += check;
    // }

    std::cout << "Z = " << Z << std::endl;

  }

  // std::cout << "detD = " << matD.determinant() << std::endl;

  // std::cout << (matD0 * matD0.adjoint()).trace() << std::endl;
  // matD0 = matD0 - matD;
  // std::cout << (matD0 * matD0.adjoint()).trace() << std::endl;

  // assert(Lx==6);
  // assert(Ly==6);

  // Eigen::MatrixXcd matDp( 2*36, 2*24 );
  // int i0=0, i=0;
  // for(int x=0; x<Lx; x++){
  //   for(int y=0; y<Ly; y++){
  //     if( is_site(x,y) ) {
  //       matDp.block(0, 2*i, 2*36, 2) = matD.block(0, 2*i0, 2*36, 2);
  //       i++;
  //     }
  //     i0++;
  //   }
  // }

  // std::cout << matDp << std::endl;

  // Eigen::MatrixXcd matDpp( 2*24, 2*24 );
  // i0=0, i=0;
  // for(int x=0; x<Lx; x++){
  //   for(int y=0; y<Ly; y++){
  //     if( is_site(x,y) ) {
  //       matDpp.block(2*i, 0, 2, 2*24) = matDp.block(2*i0, 0, 2, 2*24);
  //       i++;
  //     }
  //     i0++;
  //   }
  // }

  // std::cout << "detDpp = " << matDpp.determinant() << std::endl;
  // std::cout << matDpp << std::endl;

  return 0;
}


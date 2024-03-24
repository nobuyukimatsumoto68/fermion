#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <algorithm>

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

  const int nu = 3;

  {
    const int N = 2*Lx*Ly;

    Eigen::MatrixXcd D = get_Dirac_matrix(nu);
    // std::cout << "matD0 = " << std::endl;
    // for(int j=0; j<D.rows(); j++){
    //   for(int i=0; i<D.cols(); i++){
    //     std::cout << i << ' ' << j << ' ' << real(D(i,j)) << " " << imag(D(i,j)) << std::endl;
    //   }}

    std::vector<int> vacant;
    for(int x=0; x<Lx; x++){
      for(int y=0; y<Ly; y++){
        const Idx idx1 = 2*idx(x,y);
        if( !is_site(x,y) ) {
          vacant.push_back( idx1 );
          vacant.push_back( idx1+1 );
          // std::cout << "debug." << std::endl;
        }
      }}

    std::sort(vacant.begin(),vacant.end());

    std::cout << "vacant = " << std::endl;
    for(auto elem : vacant) std::cout << elem << " ";
    std::cout << std::endl;

    std::vector<std::complex<double>> D_removed;
    int idx_tot = 0;

    int js=0;
    for(int j=0; j<N; j++){
      // std::cout << "! j = " << j << std::endl;
      if(j==vacant[js]){
        // std::cout << "!! J = " << j << std::endl;
        js++;
        continue;
      }
      int is = 0;
      for(int i=0; i<N; i++){
        if(i==vacant[is]){
          // std::cout << "i = " << i << std::endl;
          is++;
          continue;
        }

        D_removed.push_back( D(i,j) );
        if(idx_tot>=N*N*4/9) assert(false);
        idx_tot++;
      }}
    // std::cout << "idx_tot = " << idx_tot << std::endl;

    Eigen::MatrixXcd D_removed_eigen = Eigen::Map<Eigen::MatrixXcd>(&D_removed[0], N*2/3, N*2/3);
    std::complex det0 = D_removed_eigen.determinant();
    std::cout << "det = " << det0 << std::endl;
    std::cout << "log(det) = " << log(det0) << std::endl;
  }



  {
    Eigen::MatrixXcd matD0 = get_Dirac_matrix( nu );

    Eigen::MatrixXcd matD( 2*Lx*Ly, 2*Lx*Ly );
    for(int i=0; i<2*Lx*Ly; i++){
      Vect e = Eigen::VectorXcd::Zero(2*Lx*Ly);
      e( i ) = 1.0;
      matD.block(0, i, 2*Lx*Ly, 1) = multD_eigen(e, nu);
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
  }





  {
    const double delta = 1.0e-5;

    std::complex detp = 1.0, detm = 1.0;

    {
      const double Mu = 1.0 + 0.5*delta;
      Eigen::MatrixXcd matD0 = get_Dirac_matrix( nu );
      Eigen::MatrixXcd eps = get_large_epsilon();
      Eigen::MatrixXcd A = eps*matD0;
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
      ces.compute( A );
      Eigen::VectorXcd ev = ces.eigenvalues();

      int i0=0, i=0;
      for(auto elem : ev ){
        if( std::abs(elem)<1.0e-14 ){
          i0++;
        }
        else{
          i++;
          detp *= elem;
        }
      }
    }

    {
      const double Mu = 1.0 - 0.5*delta;
      Eigen::MatrixXcd matD0 = get_Dirac_matrix( nu );
      Eigen::MatrixXcd eps = get_large_epsilon();
      Eigen::MatrixXcd A = eps*matD0;
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
      ces.compute( A );
      Eigen::VectorXcd ev = ces.eigenvalues();

      int i0=0, i=0;
      for(auto elem : ev ){
        if( std::abs(elem)<1.0e-14 ){
          i0++;
        }
        else{
          i++;
          detm *= elem;
        }
      }
    }

    std::complex<double> Pfp = std::sqrt( detp.real() );
    std::complex<double> Pfm = std::sqrt( detm.real() );
    std::cout << "Pf+ = " << Pfp << std::endl;
    std::cout << "Pf_ = " << Pfm << std::endl;
    std::cout << "(1/V) dlog[Pf] / dMu = "
              << (1.0/6.0)*(std::log(Pfp)-std::log(Pfm))/delta << std::endl;
  }




















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

  int sx=1, sy=1;
  if(nu>=3) sx = -1;
  if(nu/2==1) sy = -1;

  {
    // double Z = 1.0;
    Eigen::MatrixXcd C;
    Eigen::MatrixXcd D00 = Eigen::MatrixXcd::Zero(2,2);
    std::complex<double> tr, diff, Z;
    double check;
    int len;
    Z = 1.0;

    {
      //1
      C = R*PI*Q*RI*P*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //2
      C = R*QI*P*RI*Q*PI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //3
      C = P*RI*Q*PI*R*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //4
      C = sy*R*PI*R*QI;
      // C = R*PI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //5
      C = sy*QI*R*PI*R;
      // C = QI*R*PI*R;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      //6
      C = sy*R*QI*R*PI;
      // C = R*QI*R*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //7
      C = sx*PI*R*PI*Q;
      // C = PI*R*PI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      //8
      C = sx*P*RI*P*QI;
      // C = P*RI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //9
      C = sx*R*PI*Q*PI;
      // C = R*PI*Q*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //10
      C = sx*sy*R*QI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //11
      C = sx*sy*P*QI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //12
      C = sx*sy*PI*Q*RI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;
    }

    {
      //13
      C = sx*R*QI*R*QI*R*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //14
      C = sx*sy*R*PI*R*PI*R*PI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa,len) * tr - special_sign*check;
      // std::cout << "mtr = " << std::pow(kappa,len) * tr << std::endl
      //           << "ch = " << special_sign*check << std::endl;

      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    {
      //15
      C = sy*P*QI*P*QI*P*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa,len) * tr;

      D00 += C;
    }

    std::cout << "Z = " << Z << std::endl;
    std::cout << "D00/Z = " << D00/Z << std::endl;
  }


  {
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

    {
      std::cout << "Dinv = " << std::endl;
      int x=0, y=0;
      std::cout << Dinv_n_0[idx(x,y)] << std::endl;
    }


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


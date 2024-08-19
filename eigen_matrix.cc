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



int main(int argc, char **argv){

#ifdef _OPENMP
  omp_set_num_threads( nparallel );
#endif

  std::cout << std::scientific << std::setprecision(15) << std::endl;

  if (argc>1) nu = atoi(argv[1]);
  // if (argc>2) kappa = atoi(argv[2]);
  std::cout << "nu = " << nu << std::endl;
  // std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"theta"+std::to_string(theta);

  set_all();

  {
    const int N = 2*Lx*Ly;

    Eigen::MatrixXcd D = get_Dirac_matrix();
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
        }
      }}

    std::sort(vacant.begin(),vacant.end());

    std::vector<std::complex<double>> D_removed;
    int idx_tot = 0;

    int js=0;
    for(int j=0; j<N; j++){
      if(j==vacant[js]){
        js++;
        continue;
      }
      int is = 0;
      for(int i=0; i<N; i++){
        if(i==vacant[is]){
          is++;
          continue;
        }

        D_removed.push_back( D(i,j) );
        if(idx_tot>=N*N*4/9) assert(false);
        idx_tot++;
      }}
    // std::cout << "idx_tot = " << idx_tot << std::endl;

    Eigen::MatrixXcd D_removed_eigen = Eigen::Map<Eigen::MatrixXcd>(&D_removed[0], N*2/3, N*2/3);
    // std::cout << "Dinv = " << std::endl
    //           << D_removed_eigen.inverse() << std::endl;
    std::complex det0 = D_removed_eigen.determinant();
    std::cout << "det = " << det0 << std::endl;
    std::cout << "log(det) = " << log(det0) << std::endl;
  }


  std::complex<double> det_removed = 1.0;
  double pf_removed = 0.0;
  {
    Eigen::MatrixXcd matD0 = get_Dirac_matrix();

    Eigen::MatrixXcd matD( 2*Lx*Ly, 2*Lx*Ly );
    for(int i=0; i<2*Lx*Ly; i++){
      Vect e = Eigen::VectorXcd::Zero(2*Lx*Ly);
      e( i ) = 1.0;
      matD.block(0, i, 2*Lx*Ly, 1) = multD_eigen(e);
    }
    Eigen::MatrixXcd diff = matD - matD0;
    // std::cout << "diff; Eig-Mult = " << (diff*diff.adjoint()).trace() << std::endl;

    Eigen::MatrixXcd eps = get_large_epsilon();
    Eigen::MatrixXcd A = eps*matD;
    diff = A + A.transpose();
    // std::cout << "diff; Asymm = " << (diff*diff.adjoint()).trace() << std::endl;

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute( A );
    Eigen::VectorXcd ev = ces.eigenvalues();

    int jj = 0, kk = 0;
    std::vector<std::complex<double>> vec(ev.data(), ev.data() + ev.rows() * ev.cols());
    for(auto elem : vec) {
      if( std::abs(elem)>1.0e-14 ) {
        det_removed *= elem;
        kk++;
      }
      else jj++;
    }
    pf_removed = std::sqrt( det_removed.real() );
    std::cout << "detR = " << det_removed << std::endl
              << "PfR = " << pf_removed << std::endl;
    std::cout << "jj = " << jj << std::endl;
    std::cout << "kk = " << kk << std::endl;
  }




  // {
  //   const double delta = 1.0e-5;

  //   auto get_det = [&](const double Mu_, const int nu_){
  //     nu=nu_;
  //     // Eigen::MatrixXcd eps = get_large_epsilon();
  //     Eigen::MatrixXcd A = get_Dirac_matrix( Mu_ );
  //     Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
  //     ces.compute( A );
  //     Eigen::VectorXcd ev = ces.eigenvalues();

  //     std::complex<double> res = 1.0;
  //     for(auto elem : ev ){
  //       if( std::abs(elem)>1.0e-14 ) res *= elem;
  //       // else assert(false);
  //     }
  //     return res;
  //   };

  //   std::complex detp1 = get_det( 1.0 + 0.5*delta, 1 );
  //   std::complex detp2 = get_det( 1.0 + 0.5*delta, 2 );
  //   std::complex detp3 = get_det( 1.0 + 0.5*delta, 3 );
  //   std::complex detp4 = get_det( 1.0 + 0.5*delta, 4 );

  //   std::complex detm1 = get_det( 1.0 - 0.5*delta, 1 );
  //   std::complex detm2 = get_det( 1.0 - 0.5*delta, 2 );
  //   std::complex detm3 = get_det( 1.0 - 0.5*delta, 3 );
  //   std::complex detm4 = get_det( 1.0 - 0.5*delta, 4 );

  //   std::complex det1 = get_det( 1.0, 1 );
  //   std::complex det2 = get_det( 1.0, 2 );
  //   std::complex det3 = get_det( 1.0, 3 );
  //   std::complex det4 = get_det( 1.0, 4 );

  //   std::complex<double> Pf1p = std::sqrt( detp1.real() );
  //   std::complex<double> Pf2p = std::sqrt( detp2.real() );
  //   std::complex<double> Pf3p = std::sqrt( detp3.real() );
  //   std::complex<double> Pf4p = std::sqrt( detp4.real() );
  //   std::complex<double> Pf1m = std::sqrt( detm1.real() );
  //   std::complex<double> Pf2m = std::sqrt( detm2.real() );
  //   std::complex<double> Pf3m = std::sqrt( detm3.real() );
  //   std::complex<double> Pf4m = std::sqrt( detm4.real() );
  //   std::complex<double> Pf1 = std::sqrt( det1.real() );
  //   std::complex<double> Pf2 = std::sqrt( det2.real() );
  //   std::complex<double> Pf3 = std::sqrt( det3.real() );
  //   std::complex<double> Pf4 = std::sqrt( det4.real() );

  //   std::complex<double> Zp = (Pf1p + Pf2p + Pf3p + Pf4p)*0.5;
  //   std::complex<double> Zm = (Pf1m + Pf2m + Pf3m + Pf4m)*0.5;
  //   std::complex<double> Z = (Pf1 + Pf2 + Pf3 + Pf4)*0.5;
  //   std::cout << "Z = " << Z << std::endl;
  //   // std::cout << "dlnZ = " << Z << std::endl;
  //   // std::cout << "Z = " << Pf1 << std::endl;
  //   // std::cout << "Z = " << Pf2 << std::endl;
  //   // std::cout << "Z = " << Pf3 << std::endl;
  //   // std::cout << "Z = " << Pf4 << std::endl;
  //   // std::cout << "Pf1+ = " << Pfp << std::endl;
  //   // std::cout << "Pf1_ = " << Pfm << std::endl;
  //   std::cout << "(1/V) dlog[Z] / dMu = "
  //             << (1.0/6.0)*(std::log(Zp)-std::log(Zm))/delta << std::endl;
  // }












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



  double kappa_ = kappa[0];


  {
    // double Z = 1.0;
    Eigen::MatrixXcd C;
    Eigen::MatrixXcd D00 = Eigen::MatrixXcd::Zero(2,2);
    std::complex<double> tr, diff, Z, Z1, Z2, ZA, ZB, ZC;
    double check;
    int len;
    Z = 1.0;
    Z1 = 1.0;
    Z2 = 1.0;

    ZA = 0.0;
    ZB = 0.0;
    ZC = 0.0;

    {
      //1
      C = R*PI*Q*RI*P*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZB += std::pow(kappa_,len-1) * tr;
      ZC += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //2
      C = R*QI*P*RI*Q*PI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZC += std::pow(kappa_,len-1) * tr;
      ZA += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //3
      C = P*RI*Q*PI*R*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZA += std::pow(kappa_,len-1) * tr;
      ZB += std::pow(kappa_,len-1) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z2 += std::pow(kappa_,len) * tr;
      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //4
      C = sy*R*PI*R*QI;
      // C = R*PI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZB += std::pow(kappa_,len-1) * tr;
      ZC += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //5
      C = sy*QI*R*PI*R;
      // C = QI*R*PI*R;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z1 += std::pow(kappa_,len) * tr;
      Z2 += std::pow(kappa_,len) * tr;
    }

    {
      //6
      C = sy*R*QI*R*PI;
      // C = R*QI*R*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZA += std::pow(kappa_,len-1) * tr;
      ZC += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //7
      C = sx*PI*R*PI*Q;
      // C = PI*R*PI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z1 += std::pow(kappa_,len) * tr;
      Z2 += std::pow(kappa_,len) * tr;
    }

    {
      //8
      C = sx*P*RI*P*QI;
      // C = P*RI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZA += std::pow(kappa_,len-1) * tr;
      ZB += std::pow(kappa_,len-1) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z2 += std::pow(kappa_,len) * tr;
    }

    {
      //9
      C = sx*R*PI*Q*PI;
      // C = R*PI*Q*PI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZC += std::pow(kappa_,len-1) * tr;
      ZA += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //10
      C = sx*sy*R*QI*P*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa_,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZB += std::pow(kappa_,len-1) * tr;
      ZC += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //11
      C = sx*sy*P*QI*R*QI;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa_,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZA += std::pow(kappa_,len-1) * tr;
      ZB += std::pow(kappa_,len-1) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z2 += std::pow(kappa_,len) * tr;
    }

    {
      //12
      C = sx*sy*PI*Q*RI*Q;
      len = 4;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa_,len) * tr - special_sign*check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z1 += std::pow(kappa_,len) * tr;
      Z2 += std::pow(kappa_,len) * tr;
    }

    {
      //13
      C = sx*R*QI*R*QI*R*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZB += std::pow(kappa_,len-1) * tr;
      ZC += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //14
      C = sx*sy*R*PI*R*PI*R*PI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      int special_sign = -1.0;

      diff = std::pow(kappa_,len) * tr - special_sign*check;
      // std::cout << "mtr = " << std::pow(kappa_,len) * tr << std::endl
      //           << "ch = " << special_sign*check << std::endl;

      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZC += std::pow(kappa_,len-1) * tr;
      ZA += std::pow(kappa_,len-1) * tr;

      // D00 += std::pow(kappa_,len) * C;
    }

    {
      //15
      C = sy*P*QI*P*QI*P*QI;
      len = 6;
      tr = -C.trace();
      check = std::pow( kappa_*std::cos(M_PI/6.0), len);

      diff = std::pow(kappa_,len) * tr - check;
      // assert( std::abs(diff)<1.0e-14 );
      Z += std::pow(kappa_,len) * tr;
      ZA += std::pow(kappa_,len-1) * tr;
      ZB += std::pow(kappa_,len-1) * tr;

      D00 += std::pow(kappa_,len) * C;
      Z2 += std::pow(kappa_,len) * tr;
    }

    std::cout << "kappa_ = " << kappa_ << std::endl;
    std::cout << "Z = " << Z << std::endl;
    std::cout << "Z1 = " << Z1 << std::endl;
    std::cout << "ZA = " << ZA << std::endl;
    std::cout << "ZB = " << ZB << std::endl;
    std::cout << "ZC = " << ZC << std::endl;
    std::cout << "Z1/Z = " << Z1/Z << std::endl;
    std::cout << "ZA/Z = " << ZA/Z << std::endl;
    std::cout << "ZB/Z = " << ZB/Z << std::endl;
    std::cout << "ZC/Z = " << ZC/Z << std::endl;
    std::cout << "pf_removed = " << pf_removed << std::endl;
    std::cout << "Z1/Z' = " << Z1/pf_removed << std::endl;
    std::cout << "Z2/Z' = " << Z2/Z << std::endl;

    std::cout << "D00 = " << D00 << std::endl;
    std::cout << "D00/Z = " << D00/Z << std::endl;
  }


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
      std::cout << "Dinv = " << std::endl;
      int x=0, y=0;
      std::cout << Dinv_n_0[idx(x,y)] << std::endl;
    }

  }



  // {
  //   std::cout << "zero mode." << std::endl;
  //   Eigen::MatrixXcd matD0 = get_Dirac_matrix();
  //   Vect e = Eigen::VectorXcd::Zero(2*Lx*Ly);
  //   // for(int i=0; i<2*Lx*Ly; i++) e( i ) = 1.0;
  //   for(int i=0; i<Lx*Ly; i++) e( 2*i+1 ) = 1.0;
  //   std::cout << "De0 = " << std::endl;
  //   auto v1 = matD0 * e;
  //   std::cout << v1 << std::endl;
  //   std::cout << v1.norm() << std::endl;
  //   std::cout << "Ddag e0 = " << std::endl;
  //   std::cout << matD0.adjoint() * e << std::endl;
  // }

















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


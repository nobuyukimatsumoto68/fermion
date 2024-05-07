#pragma once

#include <complex>
#include <array>
#include <cassert>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "typedefs.hpp"
#include "constants.hpp"

// ======================================

Pauli get_Pauli();
const Pauli sigma = get_Pauli();

const Complex I = Complex(0.0, 1.0);





void set_ell(){
  ell0[0] = 1.0;
  ell0[1] = 0.0;

  ell2[0] = -omega[0];
  ell2[1] = -omega[1];

  ell1[0] = -ell2[0] - ell0[0];
  ell1[1] = -ell2[1] - ell0[1];

  ell[0] = std::sqrt( ell0[0]*ell0[0] + ell0[1]*ell0[1] );
  ell[1] = std::sqrt( ell1[0]*ell1[0] + ell1[1]*ell1[1] );
  ell[2] = std::sqrt( ell2[0]*ell2[0] + ell2[1]*ell2[1] );
}


void set_kappa(){
  kappa[0] = 2.0*ell[0] / (ell[0] + ell[1] + ell[2]);
  kappa[1] = 2.0*ell[1] / (ell[0] + ell[1] + ell[2]);
  kappa[2] = 2.0*ell[2] / (ell[0] + ell[1] + ell[2]);
}


void set_ell_star(){
  const double s = 0.5 * (ell[0]+ell[1]+ell[2]);
  const double area = std::sqrt( s*(s-ell[0])*(s-ell[1])*(s-ell[2]) );

  double coeff;
  coeff = 0.25 * (ell[1]*ell[1] + ell[2]*ell[2] - ell[0]*ell[0]) / area;
  ell_star0[1] = -ell0[1] * coeff;
  ell_star0[0] = -ell0[0] * coeff;
  // ell_star0[0] =  ell0[1] * coeff;
  // ell_star0[1] = -ell0[0] * coeff;

  coeff = 0.25 * (ell[2]*ell[2] + ell[0]*ell[0] - ell[1]*ell[1]) / area;
  ell_star1[1] = -ell1[1] * coeff;
  ell_star1[0] = -ell1[0] * coeff;
  // ell_star1[0] =  ell1[1] * coeff;
  // ell_star1[1] = -ell1[0] * coeff;

  coeff = 0.25 * (ell[0]*ell[0] + ell[1]*ell[1] - ell[2]*ell[2]) / area;
  ell_star2[1] = -ell2[1] * coeff;
  ell_star2[0] = -ell2[0] * coeff;
  // ell_star2[0] =  ell2[1] * coeff;
  // ell_star2[1] = -ell2[0] * coeff;
}


void set_e(){
  double len;

  len = std::sqrt( ell_star0[0]*ell_star0[0] + ell_star0[1]*ell_star0[1] );
  e0[0] = ell_star0[0]/len;
  e0[1] = ell_star0[1]/len;

  len = std::sqrt( ell_star1[0]*ell_star1[0] + ell_star1[1]*ell_star1[1] );
  e1[0] = ell_star1[0]/len;
  e1[1] = ell_star1[1]/len;

  len = std::sqrt( ell_star2[0]*ell_star2[0] + ell_star2[1]*ell_star2[1] );
  e2[0] = ell_star2[0]/len;
  e2[1] = ell_star2[1]/len;
}


void set_all(){
  set_ell();
  set_kappa();
  set_ell_star();
  set_e();
}


// void get_e( double* res, const int mu ){
//   if(mu==0){
//     // res[0] = -1.0;
//     // res[1] = 0.0;
//     res[0] = d_e0[0];
//     res[1] = d_e0[1];
//   }
//   else if(mu==1){
//     // res[0] = 0.5;
//     // res[1] = -0.5*sqrt(3.0);
//     res[0] = d_e1[0];
//     res[1] = d_e1[1];
//   }
//   else if(mu==2){
//     // res[0] = 0.5;
//     // res[1] = 0.5*sqrt(3.0);
//     res[0] = d_e2[0];
//     res[1] = d_e2[1];
//   }
//   else if(mu==3){
//     // res[0] = 1.0;
//     // res[1] = 0.0;
//     res[0] = -d_e0[0];
//     res[1] = -d_e0[1];
//   }
//   else if(mu==4){
//     // res[0] = -0.5;
//     // res[1] = 0.5*sqrt(3.0);
//     res[0] = -d_e1[0];
//     res[1] = -d_e1[1];
//   }
//   else if(mu==5){
//     // res[0] = -0.5;
//     // res[1] = -0.5*sqrt(3.0);
//     res[0] = -d_e2[0];
//     res[1] = -d_e2[1];
//   }
//   else assert(false);
// }


int mod(const int a, const int b){ return (b +(a%b))%b; }

bool is_site(const int x, const int y) {
  const int c = mod(x-y, 3);

  bool res = true;
  if(c==1) res = false; // e.g., (1,0)

  return res;
}


bool is_link(const int x, const int y, const int mu) {
  const int c = mod(x-y, 3);

  bool res = false;
  if(c==0 && mu<3) res = true; // e.g., (0,0)
  else if(c==2 && mu>=3) res = true; // e.g., (0,1)

  return res;
}


Pauli get_Pauli() {
  Pauli sigma;
  sigma[0] << 1,0,0,1;
  sigma[1] << 0,1,1,0;
  sigma[2] << 0,-I,I,0;
  sigma[3] << 1,0,0,-1;
  return sigma;
}


// Idx idx(const int x, const int y){ return x + Lx*y; }
Idx idx(const int x, const int y){ return (x+Lx)%Lx + Lx * ((y+Ly)%Ly); }


int cshift(int& xp, int& yp, const int x, const int y, const int mu){
  int res = 1;

  if(mu==0){
    xp=mod(x-1,Lx);
    yp=y;

    if(x==0 && nu>=3) res *= -1;
  }
  else if(mu==1){
    xp=mod(x+1,Lx);
    yp=mod(y-1,Ly);

    if(x==Lx-1 && nu>=3) res *= -1;
    if(y==0 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(xp-Ly/2<0) res *= -1;
        xp=mod(xp-int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==2){
    xp=x;
    yp=mod(y+1,Ly);

    if(y==Ly-1 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(Lx<=xp+Ly/2) res *= -1;
        xp=mod(xp+int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==3){
    xp=mod(x+1,Lx);
    yp=y;

    if(x==Lx-1 && nu>=3) res *= -1;
  }
  else if(mu==4){
    xp=mod(x-1,Lx);
    yp=mod(y+1,Ly);

    if(x==0 && nu>=3) res *= -1;
    if(y==Ly-1 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(Lx<=xp+Ly/2) res *= -1;
        xp=mod(xp+int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==5){
    xp=x;
    yp=mod(y-1,Ly);

    if(y==0 && nu/2==1 ) {
      if(is_periodic_orthogonal) {
        if(xp-Ly/2<0) res *= -1;
        xp=mod(xp-int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else assert(false);
  return res;
}


int cshift_minus(int& xp, int& yp, const int x, const int y, const int mu){
  int res = 1;

  if(mu==0){
    xp=mod(x+1,Lx);
    yp=y;

    if(x==Lx-1 && nu>=3) res *= -1;
  }
  else if(mu==1){
    xp=mod(x-1,Lx);
    yp=mod(y+1,Ly);

    if(x==0 && nu>=3) res *= -1;
    if(y==Ly-1 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(Lx<=xp+Ly/2) res *= -1;
        xp=mod(xp+int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==2){
    xp=x;
    yp=mod(y-1,Ly);

    if(y==0 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(xp-Ly/2<0) res *= -1;
        xp=mod(xp-int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==3){
    xp=mod(x-1,Lx);
    yp=y;

    if(x==0 && nu>=3) res *= -1;
  }
  else if(mu==4){
    xp=mod(x+1,Lx);
    yp=mod(y-1,Ly);

    if(x==Lx-1 && nu>=3) res *= -1;
    if(y==0 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(xp-Ly/2<0) res *= -1;
        xp=mod(xp-int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else if(mu==5){
    xp=x;
    yp=mod(y+1,Ly);

    if(y==Ly-1 && nu/2==1) {
      if(is_periodic_orthogonal) {
        if(Lx<=xp+Ly/2) res *= -1;
        xp=mod(xp+int(Ly/2),Lx);
      }
      res *= -1;
    }
  }
  else assert(false);

  return res;
}



V2 get_e( const int mu ){
  V2 res;

  if(mu==0){
    // res(0) = -1.0;
    // res(1) = 0.0;
    res(0) = e0[0];
    res(1) = e0[1];
  }
  else if(mu==1){
    // res(0) = 0.5;
    // res(1) = -0.5*sqrt(3);
    res(0) = e1[0];
    res(1) = e1[1];
  }
  else if(mu==2){
    // res(0) = 0.5;
    // res(1) = 0.5*sqrt(3);
    res(0) = e2[0];
    res(1) = e2[1];
  }
  else if(mu==3){
    // res(0) = 1.0;
    // res(1) = 0.0;
    res(0) = -e0[0];
    res(1) = -e0[1];
  }
  else if(mu==4){
    // res(0) = -0.5;
    // res(1) = 0.5*sqrt(3);
    res(0) = -e1[0];
    res(1) = -e1[1];
  }
  else if(mu==5){
    // res(0) = -0.5;
    // res(1) = -0.5*sqrt(3);
    res(0) = -e2[0];
    res(1) = -e2[1];
  }
  else assert(false);

  return res;
}


M2 get_eps(){
  M2 res;
  res << 0, 1, -1, 0;
  return res;
}


M2 get_gamma( const int mu ){
  V2 e = get_e( mu );
  M2 res = e(0)*sigma[1] + e(1)*sigma[2];
  return res;
}

M2 Wilson_projector( const int mu ){
  return 0.5 * ( sigma[0] - get_gamma(mu) );
}


Eigen::MatrixXcd get_Dirac_matrix ( const double Mu=1.0 ){ //
  Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(TWO*Lx*Ly, TWO*Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      if( is_site(x,y) ) res.block<2,2>(2*idx(x,y), 2*idx(x,y)) = Mu*sigma[0];
    }
  }

  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y,mu) ) {
          int xp, yp;
          const int sign = cshift( xp, yp, x, y, mu );
          const Idx idx1 = 2*idx(x,y);
          const Idx idx2 = 2*idx(xp,yp);
          res.block<2,2>(idx1, idx2) = -sign * kappa[mu%3] * Wilson_projector(mu);
        }
      }
    }

  return res;
}


Eigen::MatrixXcd get_large_epsilon (){
  Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(TWO*Lx*Ly, TWO*Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      if( is_site(x,y) ) res.block<2,2>(2*idx(x,y), 2*idx(x,y)) = I*sigma[2];
    }
  }

  return res;
}



Eigen::VectorXcd multD_eigen ( const Eigen::VectorXcd& v){
  Eigen::VectorXcd res = Eigen::VectorXcd::Zero(2*Lx*Ly);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      const Idx idx1 = 2*idx(x,y);
      if( is_site(x,y) ) res.segment(idx1,2) += v.segment(idx1,2);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y,mu) ) {
          int xp, yp;
          const int sign = cshift( xp, yp, x, y, mu );
          const Idx idx1 = 2*idx(x,y);
          const Idx idx2 = 2*idx(xp,yp);
          res.segment(idx1, 2) -= sign * kappa[mu%3] * Wilson_projector(mu) * v.segment(idx2, 2);
        }
      }
    }

  return res;
}


Eigen::VectorXcd multDdagger_eigen ( const Eigen::VectorXcd& v ){
  Eigen::VectorXcd res = Eigen::VectorXcd::Zero(2*Lx*Ly);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      const Idx idx1 = 2*idx(x,y);
      if( is_site(x,y) ) res.segment(idx1,2) += v.segment(idx1,2);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y, (mu+THREE)%SIX) ) {
          int xp, yp;
          const int sign = cshift_minus( xp, yp, x, y, mu );
          const Idx idx1 = 2*idx(x,y);
          const Idx idx2 = 2*idx(xp,yp);
          res.segment(idx1, 2) -= sign * kappa[mu%3] * Wilson_projector(mu) * v.segment(idx2, 2);
        }
      }
    }

  return res;
}




Vect A(const Vect& v){
  Vect res = multD_eigen(v);
  res = multDdagger_eigen(res);
  return res;
}


Vect CG(const Vect& init,
        const Vect& b,
        const double TOL=1.0e-13,
        const int MAXITER=1e5
        ){
  Vect x = init; //Vect::Zero(ops.size);
  Vect r = b; // b - A(x);
  Vect p = r;

  double mu = r.squaredNorm();
  double mu_old = mu;
  const double b_norm_sq = b.squaredNorm();
  const double mu_crit = TOL*TOL*b_norm_sq;

  if(mu<mu_crit){
    std::clog << "NO SOLVE" << std::endl;
  }
  else{
    int k=0;
    for(; k<MAXITER; ++k){
      const Vect q = A(p);
      const Complex gam = p.dot(q);
      const Complex al = mu/gam;
      // std::cout << "debug. iter = " << k << std::endl;
      // std::cout << "debug. x = " << x.transpose() << std::endl;
      // std::cout << "debug. p = " << p.transpose() << std::endl;
      x += al*p;
      r -= al*q;
      mu = r.squaredNorm();
      // std::clog << "mu = " << mu << std::endl;
      if(mu<mu_crit) break;
      const double bet = mu/mu_old;
      mu_old = mu;
      p = r+bet*p;
    }
    std::clog << "SOLVER:       #iterations: " << k << std::endl;
  }

  std::clog << "error1: " << std::sqrt(mu/b_norm_sq) << std::endl
            << "error2: " << std::sqrt(r.squaredNorm()/b_norm_sq) << std::endl;

  return x;
}


#include <iostream>
#include <iomanip>
#include <complex>
#include <array>
#include <cassert>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


// ======================================


using Complex = std::complex<double>;
using M2 = Eigen::Matrix2cd;
using V2 = Eigen::Vector2cd;

using Vect = Eigen::VectorXcd;
using Mat = Eigen::MatrixXcd;

using Pauli = std::array<M2, 4>;


// ======================================


constexpr int TWO = 2;
constexpr int SIX = 6;
constexpr Complex I = Complex(0, 1);


const int Lx = 6 * 4;
const int Ly = 6 * 4;
const double alat = 0.1; // ell

const double m = 0.01;

// const double Vy = 3.0*sqrt(3.0)/4.0 * alat*alat;
const double my = m + 2.0/3.0 * 3.0/alat;
const double kappa = (2.0/3.0) * 2.0 / alat / my;


Pauli get_Pauli();
const Pauli sigma = get_Pauli();

// ======================================


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
  sigma[1] << 0,-I,I,0;
  sigma[2] << 0,1,1,0;
  sigma[3] << 1,0,0,-1;
  return sigma;
}


long int idx(const int x, const int y){ return x + Lx*y; }


int cshift(int& xp, int& yp, const int x, const int y, const int mu){
  int res = 1;

  if(mu==0){
    xp=mod(x-1,Lx);
    yp=y;

    if(x==0) res *= -1;
  }
  else if(mu==1){
    xp=mod(x+1,Lx);
    yp=mod(y-1,Ly);

    if(x==Lx-1) res *= -1;
    if(y==0) {
      // xp=mod(xp-int(Ly/2),Lx);
      res *= -1;
    }
  }
  else if(mu==2){
    xp=x;
    yp=mod(y+1,Ly);

    if(y==Ly-1) {
      // xp=mod(xp+int(Ly/2),Lx);
      res *= -1;
    }
  }
  else if(mu==3){
    xp=mod(x+1,Lx);
    yp=y;

    if(x==Lx-1) res *= -1;
  }
  else if(mu==4){
    xp=mod(x-1,Lx);
    yp=mod(y+1,Ly);

    if(x==0) res *= -1;
    if(y==Ly-1) {
      // xp=mod(xp+int(Ly/2),Lx);
      res *= -1;
    }
  }
  else if(mu==5){
    xp=x;
    yp=mod(y-1,Ly);

    if(y==0) {
      // xp=mod(xp-int(Ly/2),Lx);
      res *= -1;
    }
  }
  else assert(false);

  return res;
}


V2 get_e( const int mu ){
  V2 res;

  if(mu==0){
    res(0) = -1.0;
    res(1) = 0.0;
  }
  else if(mu==1){
    res(0) = 0.5;
    res(1) = -0.5*sqrt(3);
  }
  else if(mu==2){
    res(0) = 0.5;
    res(1) = 0.5*sqrt(3);
  }
  else if(mu==3){
    res(0) = 1.0;
    res(1) = 0.0;
  }
  else if(mu==4){
    res(0) = -0.5;
    res(1) = 0.5*sqrt(3);
  }
  else if(mu==5){
    res(0) = -0.5;
    res(1) = -0.5*sqrt(3);
  }
  else assert(false);

  return res;
}


M2 Wilson_projector( const int mu ){
  V2 e = get_e(mu);

  M2 res = -e(0)*sigma[2] - e(1)*sigma[1];
  res += sigma[0];
  res *= 0.5;
  return res;
}


Eigen::MatrixXcd get_Dirac_matrix (){
  Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(TWO*Lx*Ly, TWO*Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      if( is_site(x,y) ) res.block<2,2>(2*idx(x,y), 2*idx(x,y)) = 0.5*sigma[0];
    }
  }

  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y,mu) ) {
          int xp, yp;
          const int sign = cshift( xp, yp, x, y, mu );
          const long int idx1 = 2*idx(x,y);
          const long int idx2 = 2*idx(xp,yp);
          res.block<2,2>(idx1, idx2) = sign * 0.5 * kappa * Wilson_projector(mu);
        }
      }
    }

  return res;
}


Eigen::VectorXcd multD_eigen ( const Eigen::VectorXcd& v ){
  Eigen::VectorXcd res = Eigen::VectorXcd::Zero(2*Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      const long int idx1 = 2*idx(x,y);
      if( is_site(x,y) ) res.segment(idx1,2) += 0.5 * v.segment(idx1,2);
    }
  }

  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y,mu) ) {
          int xp, yp;
          const int sign = cshift( xp, yp, x, y, mu );
          const long int idx1 = 2*idx(x,y);
          const long int idx2 = 2*idx(xp,yp);
          res.segment(idx1, 2) += sign * 0.5 * kappa * Wilson_projector(mu) * v.segment(idx2, 2);
        }
      }
    }

  return res;
}


Eigen::VectorXcd multDdagger_eigen ( const Eigen::VectorXcd& v){
  Eigen::VectorXcd res = Eigen::VectorXcd::Zero(2*Lx*Ly);

  for(int x=0; x<Lx; x++){
    for(int y=0; y<Ly; y++){
      const long int idx1 = 2*idx(x,y);
      if( is_site(x,y) ) res.segment(idx1,2) += 0.5 * v.segment(idx1,2);
    }
  }

  for(int y=0; y<Ly; y++) for(int x=0; x<Lx; x++) {
      for(int mu=0; mu<SIX; mu++){
        if( is_link(x,y,mu) ) {
          int xp, yp;
          const int sign = cshift( xp, yp, x, y, mu );
          const long int idx1 = 2*idx(x,y);
          const long int idx2 = 2*idx(xp,yp);
          res.segment(idx2, 2) += sign * 0.5 * kappa * Wilson_projector(mu).adjoint() * v.segment(idx1, 2);
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
        const double TOL=1.0e-15,
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
      x += al*p;
      r -= al*q;
      mu = r.squaredNorm();
      std::clog << "mu = " << mu << std::endl;
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






// ======================================



int main(){

  // const int nthr=4;
  // omp_set_num_threads(nthr);
  // Eigen::setNbThreads(nthr);


  Eigen::MatrixXcd dirac = get_Dirac_matrix();

  // Eigen::ComplexEigenSolver<Eigen::MatrixXcd> esolver( dirac, false );
  // Eigen::VectorXcd evec = esolver.eigenvalues();
  // for(auto elem : evec){
  //   std::cout << elem.real() << " " << elem.imag() << std::endl;
  // }

  Eigen::VectorXcd vector1 = Eigen::VectorXcd::Random(2*Lx*Ly);
  Eigen::VectorXcd vector2 = vector1;

  std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

  vector1 = dirac*vector1;
  vector2 = multD_eigen(vector2);
  std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

  vector1 = dirac.adjoint() * vector1;
  vector2 = multDdagger_eigen(vector2);

  std::cout << "diff = " << (vector1-vector2).norm() << std::endl;

  Eigen::VectorXcd init = Eigen::VectorXcd::Zero(2*Lx*Ly);
  Eigen::VectorXcd b0 = Eigen::VectorXcd::Random(2*Lx*Ly);
  Vect b = multDdagger_eigen(b0);

  Vect sol = CG(init, b);

  Vect test = A(sol) - b;
  std::cout << "test = " << test.norm() << std::endl;

  Eigen::FullPivLU<Eigen::MatrixXcd> lu(dirac);
  Vect sol_direct = lu.solve(b0);
  std::cout << "solve_diff" << (sol-sol_direct).norm() << std::endl;

  // Eigen::MatrixXcd test1 = dirac + dirac.adjoint();
  // std::cout << "test1 = " << test1.norm() << std::endl;
  // Eigen::MatrixXcd test2 = dirac - dirac.adjoint();
  // std::cout << "test2 = " << test2.norm() << std::endl;

  return 0;
}











// // ======== test is_site ===============
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     std::cout << is_site(x, y);
//   }
//   std::cout << std::endl;
// }




// // ======== test is_link ===============
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     std::cout << '[';
//     for(int mu=0; mu<SIX; mu++){
//       if(mu!=0) std::cout << ' ';
//       std::cout << is_link(x, y, mu);
//     }
//     std::cout << ']';
//   }
//   std::cout << std::endl;
// }



// // ======== test get_Pauli ===============
// Pauli sigma = get_Pauli();

// sigma[0] << 1,0,0,1;
// sigma[1] << 0,-I,I,0;
// sigma[2] << 0,1,1,0;
// sigma[3] << 1,0,0,-1;

// std::cout << sigma[0] << std::endl;
// std::cout << sigma[1] << std::endl;
// std::cout << sigma[2] << std::endl;
// std::cout << sigma[3] << std::endl;



// // ======== test cshift ===============
// const int nu=5;
// for(int y=Ly-1; y>=0; y--){
//   for(int x=0; x<Lx; x++){
//     if(x!=0) std::cout << ' ';
//     int xp=0, yp=0;
//     cshift(xp, yp, x, y, nu);
//     std::cout << "(" << xp << " " << yp << ")";
//   }
//   std::cout << std::endl;
// }


// // ======== test get_e ===============
// for(int mu=0; mu<6; mu++){
//   std::cout << "mu = " << mu << std::endl;
//   std::cout << get_e( mu ) << std::endl;
// }


// // ======== test Wilson_projector ===============
// for(int mu=0; mu<6; mu++){
//   std::cout << "mu = " << mu << std::endl;
//   std::cout << Wilson_projector( mu ) << std::endl;
// }

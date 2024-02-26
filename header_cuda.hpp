#pragma once

#include <complex>
#include <array>
#include <cassert>

#include "typedefs_cuda.hpp"
#include "constants.hpp"


// ======================================


#define N (2*Lx*Ly)
#define M (Lx*Ly)

#define NThreadsPerBlock (512) // 1024
#define NBlocks (N+NThreadsPerBlock)/NThreadsPerBlock
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define D2D (cudaMemcpyDeviceToDevice)
#define DB (sizeof(double))
#define CD (sizeof(Complex))
#define cuI ( make_cuDoubleComplex(0.0,1.0) )


__device__ __host__
Complex cplx(const double c) { return make_cuDoubleComplex(c, 0.0); }


__device__ __host__ Complex  operator+(Complex a, Complex b) { return cuCadd(a,b); }
__device__ __host__ Complex  operator+(Complex a, double b) { return cuCadd(a,cplx(b)); }
__device__ __host__ Complex  operator+(double a, Complex b) { return cuCadd(cplx(a),b); }
__device__ __host__ Complex  operator-(Complex a, Complex b) { return cuCsub(a,b); }
__device__ __host__ Complex  operator-(Complex a, double b) { return cuCsub(a,cplx(b)); }
__device__ __host__ Complex  operator-(double a, Complex b) { return cuCsub(cplx(a),b); }
__device__ __host__ Complex  operator-(Complex b) { return cplx(0.0)-b; }
__device__ __host__ Complex  operator*(Complex a, Complex b) { return cuCmul(a,b); }
__device__ __host__ Complex  operator*(Complex a, double b) { return cuCmul(a,cplx(b)); }
__device__ __host__ Complex  operator*(double a, Complex b) { return cuCmul(cplx(a),b); }
__device__ __host__ Complex  operator/(Complex a, Complex b) { return cuCdiv(a,b); }
__device__ __host__ Complex  operator/(Complex a, double b) { return cuCdiv(a,cplx(b)); }
__device__ __host__ Complex  operator/(double a, Complex b) { return cuCdiv(cplx(a),b); }


__device__ __host__ inline double real(const Complex c ){ return cuCreal(c); }
__device__ __host__ inline double imag(const Complex c ){ return cuCimag(c); }
__device__ __host__ inline Complex conj(Complex c ){ return cuConj(c); }




__host__
void cudacheck( cudaError status ){
  if(status!=0) std::cout << status << std::endl;
  assert(cudaSuccess == status);
}


__host__
void set2zero( double* v, const Idx size ){ for(Idx i=0; i<size; i++) v[i] = 0.0; }
__host__
void set2zero( Complex* v, const Idx size ){ for(Idx i=0; i<size; i++) v[i] = cplx(0.0); }


__global__
void set_zero(double* d_v){
  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) d_v[i] = 0.0;
}
__global__
void set_zero(Complex* d_v){
  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) d_v[i] = cplx(0.0);
}


__host__ __device__
Idx idx(const int x, const int y){ return x + Lx*y; }


__host__ __device__
void get_xy(int& x, int& y, const Idx i){
  x = i%Lx;
  y = (i-x)/Lx;
  assert( i == idx(x,y) );
}


__host__ __device__
int mod(const int a, const int b){ return (b +(a%b))%b; }


__host__ __device__
bool is_site(const int x, const int y) {
  const int c = mod(x-y, 3);
  bool res = true;
  if(c==1) res = false; // e.g., (1,0)
  return res;
}


__host__ __device__
int is_link(const int x, const int y, const int mu) { // return c or -1
  int c = mod(x-y, 3);
  bool res = false;
  if(c==0 && mu<3) res = true; // e.g., (0,0)
  else if(c==2 && mu>=3) res = true; // e.g., (0,1)
  // if (!res) c = -1;
  return res;
}


__host__ __device__
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

    if(y==Ly-1) {
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

    if(x==Lx-1) res *= -1;
  }
  else if(mu==4){
    xp=mod(x-1,Lx);
    yp=mod(y+1,Ly);

    if(x==0) res *= -1;
    if(y==Ly-1) {
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

    if(y==0) {
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


__host__  __device__
int cshift_minus(int& xp, int& yp, const int x, const int y, const int mu){
  int res = 1;

  if(mu==0){
    xp=mod(x+1,Lx);
    yp=y;

    if(x==Lx-1) res *= -1;
  }
  else if(mu==1){
    xp=mod(x-1,Lx);
    yp=mod(y+1,Ly);

    if(x==0) res *= -1;
    if(y==Ly-1) {
      // if(is_periodic_orthogonal) {
      //   if(Lx<=xp+Ly/2) res *= -1;
      //   xp=mod(xp+int(Ly/2),Lx);
      // }
      res *= -1;
    }
  }
  else if(mu==2){
    xp=x;
    yp=mod(y-1,Ly);

    if(y==0) {
      // if(is_periodic_orthogonal) {
      //   if(xp-Ly/2<0) res *= -1;
      //   xp=mod(xp-int(Ly/2),Lx);
      // }
      res *= -1;
    }
  }
  else if(mu==3){
    xp=mod(x-1,Lx);
    yp=y;

    if(x==0) res *= -1;
  }
  else if(mu==4){
    xp=mod(x+1,Lx);
    yp=mod(y-1,Ly);

    if(x==Lx-1) res *= -1;
    if(y==0) {
      // if(is_periodic_orthogonal) {
      //   if(xp-Ly/2<0) res *= -1;
      //   xp=mod(xp-int(Ly/2),Lx);
      // }
      res *= -1;
    }
  }
  else if(mu==5){
    xp=x;
    yp=mod(y+1,Ly);

    if(y==Ly-1) {
      // if(is_periodic_orthogonal) {
      //   if(Lx<=xp+Ly/2) res *= -1;
      //   xp=mod(xp+int(Ly/2),Lx);
      // }
      res *= -1;
    }
  }
  else assert(false);

  return res;
}


__device__
void get_e( double* res, const int mu ){
  if(mu==0){
    res[0] = -1.0;
    res[1] = 0.0;
  }
  else if(mu==1){
    res[0] = 0.5;
    res[1] = -0.5*sqrt(3.0);
  }
  else if(mu==2){
    res[0] = 0.5;
    res[1] = 0.5*sqrt(3.0);
  }
  else if(mu==3){
    res[0] = 1.0;
    res[1] = 0.0;
  }
  else if(mu==4){
    res[0] = -0.5;
    res[1] = 0.5*sqrt(3.0);
  }
  else if(mu==5){
    res[0] = -0.5;
    res[1] = -0.5*sqrt(3.0);
  }
  else assert(false);
}



__global__
void daxpy(Complex* d_res, Complex* d_a, Complex* d_x, Complex* d_y){
  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) d_res[i] = *d_a * d_x[i] + d_y[i];
}



__device__
void Wilson_projector( Complex* res, const int mu, const int sign = +1 ){
  double e[2];
  get_e(e, mu);

  res[0] = cplx(0.5);
  res[3] = cplx(0.5);

  res[1] = - sign * 0.5 * ( e[0]-cuI*e[1] );
  res[2] = - sign * 0.5 * ( e[0]+cuI*e[1] );
  // res[1] = 0.5 * ( e[0]-cuI*e[1] );
  // res[2] = 0.5 * ( e[0]+cuI*e[1] );
}



__global__
void multD ( Complex* res, const Complex* v ){
  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M) {
    int x,y;
    get_xy(x,y,i);

    if( is_site(x,y) ){
      res[2*i] = res[2*i] + 0.5*v[2*i];
      res[2*i+1] = res[2*i+1] + 0.5*v[2*i+1];
    }

    for(int mu=0; mu<SIX; mu++){
      // const int c = is_link(x,y,mu);
      if( is_link(x,y,mu) ) {
        int xp, yp;
        const int sign = cshift( xp, yp, x, y, mu );
        const Idx idx2 = 2*idx(xp,yp);

        Complex P[4];
        Wilson_projector( P, mu );

        res[2*i] = res[2*i] - sign * 0.5 * kappa * (P[0]*v[idx2] + P[1]*v[idx2+1]);
        res[2*i+1] = res[2*i+1] - sign * 0.5 * kappa * (P[2]*v[idx2] + P[3]*v[idx2+1]);
      }
    }
  }
}



__global__
void multDdagger ( Complex* res, const Complex* v ){
  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M){
    int x,y;
    get_xy(x,y,i);

    if( is_site(x,y) ){
      res[2*i] = res[2*i] + 0.5*v[2*i];
      res[2*i+1] = res[2*i+1] + 0.5*v[2*i+1];
    }

    for(int mu=0; mu<SIX; mu++){
      if( is_link(x,y, (mu+THREE)%SIX) ) {
        int xp, yp;
        const int sign = cshift_minus( xp, yp, x, y, mu );
        const Idx idx2 = 2*idx(xp,yp);

        Complex P[4];
        Wilson_projector( P, mu );

        res[2*i] = res[2*i] - sign * 0.5 * kappa * (P[0]*v[idx2] + P[1]*v[idx2+1]);
        res[2*i+1] = res[2*i+1] - sign * 0.5 * kappa * (P[2]*v[idx2] + P[3]*v[idx2+1]);
      }
    }
  }
}



__host__
void multDdagger_wrapper(Complex* v, Complex* v0){
  Complex *d_v, *d_v0;

  cudacheck(cudaMalloc(&d_v, N*CD));
  cudacheck(cudaMalloc(&d_v0, N*CD));

  cudacheck(cudaMemcpy(d_v0, v0, N*CD, H2D));

  set_zero<<<NBlocks, NThreadsPerBlock>>>(d_v);
  multDdagger<<<NBlocks, NThreadsPerBlock>>>(d_v, d_v0);

  cudacheck(cudaMemcpy(v, d_v, N*CD, D2H));

  cudacheck(cudaFree(d_v));
  cudacheck(cudaFree(d_v0));
}


__host__
void multD_wrapper(Complex* v, Complex* v0){
  Complex *d_v, *d_v0;

  cudacheck(cudaMalloc(&d_v, N*CD));
  cudacheck(cudaMalloc(&d_v0, N*CD));

  cudacheck(cudaMemcpy(d_v0, v0, N*CD, H2D));

  set_zero<<<NBlocks, NThreadsPerBlock>>>(d_v);
  multD<<<NBlocks, NThreadsPerBlock>>>(d_v, d_v0);

  cudacheck(cudaMemcpy(v, d_v, N*CD, D2H));

  cudacheck(cudaFree(d_v));
  cudacheck(cudaFree(d_v0));
}




__host__
void multA(Complex* d_v, Complex* d_tmp, Complex* d_v0){
  set_zero<<<NBlocks, NThreadsPerBlock>>>(d_tmp);
  multD<<<NBlocks, NThreadsPerBlock>>>(d_tmp, d_v0);

  set_zero<<<NBlocks, NThreadsPerBlock>>>(d_v);
  multDdagger<<<NBlocks, NThreadsPerBlock>>>(d_v, d_tmp);
}


// https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
__device__
void atomicAddComplex(Complex* a, Complex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAdd(x, real(b));
  atomicAdd(y, imag(b));
}


__global__
void dot_normalized(Complex* d_res, Complex* d_p, Complex* d_q){
  __shared__ Complex tmp[NThreadsPerBlock];
  if (threadIdx.x == 0) for(int j=0; j<NThreadsPerBlock; j++) tmp[j] = cplx(0.0);
  __syncthreads();

  Idx i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    tmp[threadIdx.x] = conj(d_p[i])*d_q[i]/N;
    __syncthreads();

    if(threadIdx.x == 0){
      Complex sum = cplx(0.0);
      for(int j=0; j<NThreadsPerBlock; j++) sum = sum+tmp[j];
      atomicAddComplex(d_res, sum);
    }
  }
}




__host__
void dot_normalized_wrapper(Complex& scalar, Complex* d_scalar, Complex* d_p, Complex* d_q){
  scalar = cplx(0.0);
  cudacheck(cudaMemcpy(d_scalar, &scalar, CD, H2D));
  dot_normalized<<<NBlocks, NThreadsPerBlock>>>(d_scalar, d_p, d_q);
  cudacheck(cudaMemcpy(&scalar, d_scalar, CD, D2H));
}

__host__
void dot2self_normalized_wrapper(double& scalar, Complex* d_scalar, Complex* d_p){
  scalar = 0.0;
  Complex dummy = cplx(scalar);
  cudacheck(cudaMemcpy(d_scalar, &dummy, CD, H2D));
  dot_normalized<<<NBlocks, NThreadsPerBlock>>>(d_scalar, d_p, d_p);
  cudacheck(cudaMemcpy(&dummy, d_scalar, CD, D2H));
  assert( abs( imag(dummy)<1.0e-14 ) );
  scalar = real(dummy);
}



__host__
void solve(Complex x[N], Complex b[N], const double tol=1.0e-15, const int maxiter=1e5){
  Complex *d_x, *d_r, *d_p, *d_q, *d_tmp;
  cudacheck(cudaMalloc(&d_x, N*CD));
  cudacheck(cudaMalloc(&d_r, N*CD));
  cudacheck(cudaMalloc(&d_p, N*CD));
  cudacheck(cudaMalloc(&d_q, N*CD));
  cudacheck(cudaMalloc(&d_tmp, N*CD));

  Complex *d_scalar;
  cudacheck(cudaMalloc(&d_scalar, CD));

  set_zero<<<NBlocks, NThreadsPerBlock>>>(d_x);
  cudacheck(cudaMemcpy(d_r, b, N*CD, H2D));
  cudacheck(cudaMemcpy(d_p, d_r, N*CD, D2D));

  double mu; dot2self_normalized_wrapper(mu, d_scalar, d_r);
  assert(mu>=0.0);
  double mu_old = mu;

  double b_norm_sq; dot2self_normalized_wrapper(b_norm_sq, d_scalar, d_r);
  assert(b_norm_sq>=0.0);
  double mu_crit = tol*tol*b_norm_sq;

  if(mu<mu_crit) std::clog << "NO SOLVE" << std::endl;
  else{
    int k=0;
    Complex gam;

    for(; k<maxiter; ++k){
      multA(d_q, d_tmp, d_p);

      dot_normalized_wrapper(gam, d_scalar, d_p, d_q);

      Complex al = mu/gam;
      cudacheck(cudaMemcpy(d_scalar, &al, CD, H2D));
      daxpy<<<NBlocks, NThreadsPerBlock>>>(d_x, d_scalar, d_p, d_x);

      al = -al;
      cudacheck(cudaMemcpy(d_scalar, &al, CD, H2D));
      daxpy<<<NBlocks, NThreadsPerBlock>>>(d_r, d_scalar, d_q, d_r);

      dot2self_normalized_wrapper(mu, d_scalar, d_r);
      assert(mu>=0.0);

      if(mu<mu_crit || std::isnan(mu)) break;
      Complex bet = cplx(mu/mu_old);
      mu_old = mu;

      cudacheck(cudaMemcpy(d_scalar, &bet, CD, H2D));
      daxpy<<<NBlocks, NThreadsPerBlock>>>(d_p, d_scalar, d_p, d_r);

      if(k%100==0) {
        std::clog << "SOLVER:       #iterations: " << k << ", mu =         " << mu << std::endl;
      }
    }
    std::clog << "SOLVER:       #iterations: " << k << std::endl;
    std::clog << "SOLVER:       mu =         " << mu << std::endl;
  }

  cudacheck(cudaMemcpy(x, d_x, N*CD, D2H));

  cudacheck(cudaFree(d_x));
  cudacheck(cudaFree(d_r));
  cudacheck(cudaFree(d_p));
  cudacheck(cudaFree(d_q));
  cudacheck(cudaFree(d_tmp));
  cudacheck(cudaFree(d_scalar));
}


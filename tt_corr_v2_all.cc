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

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads( nparallel );
#endif

  if (argc>1){
    nu = atoi(argv[1]);
    // printf("%s\n", argv[i]);
  }


  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"length"+std::to_string(length)+"theta"+std::to_string(theta);
  // set_all();
  set_all();
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2);
  // const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2);
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu)+"tautil"+std::to_string(tautil1)+"_"+std::to_string(tautil2)+str(is_periodic_orthogonal);


  std::vector<M2> Dinv_n_0(Lx*Ly), Dinv_n_A(Lx*Ly), Dinv_n_B(Lx*Ly), Dinv_n_C(Lx*Ly);

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
      std::ifstream ifs( dir_data+description+"Dinv_m1_0_0.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_m1_0_1.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_1_m1_0.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_1_m1_1.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_0_1_0.dat",
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
      std::ifstream ifs( dir_data+description+"Dinv_0_1_1.dat",
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

  M2 eps = get_eps();
  M2 eps_inv = -eps;

  std::array<M2, 6> WilsonP;
  {
    for(int b=0; b<SIX; b++){
      WilsonP[b] = Wilson_projector(b);
    }
  }


  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){
      for(const int s0 : {0,1,2}){

        int dum1,dum2;
        int sign1 = cshift(dum1, dum2, 0, 0, n);

        std::vector<M2> Dinv_n_0pn;
        if(n==0) Dinv_n_0pn = Dinv_n_A;
        else if(n==1) Dinv_n_0pn = Dinv_n_B;
        else if(n==2) Dinv_n_0pn = Dinv_n_C;
        else assert(false);

        std::ofstream of( dir_data+description+"K"+std::to_string(n)+"K"+std::to_string(s0)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        for(int y=0; y<Ly; y++){
          for(int x=0; x<Lx; x++){
            int s = s0;
            if(!is_site(x,y)) continue;
            // if(!is_link(x,y,s)) s+=3;
            // if(!is_link(x,y,s)) continue;
            const int ch = mod(x-y, 3);
            // assert(ch==1);
            if(ch!=0) continue;

            const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
            const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

            int xps, yps;
            int sign2 = cshift(xps, yps, x, y, s);
            const M2& Dinv_xps_0 = sign2 * Dinv_n_0[idx(xps,yps)];
            const M2& Dinv_xps_0pn = sign1 * sign2 * Dinv_n_0pn[idx(xps,yps)];

            const Complex tr1 = ( Dinv_x_0pn * WilsonP[n+3] * eps_inv * Dinv_xps_0.transpose() * eps * WilsonP[(s+3)%6] ).trace();
            const Complex tr2 = -( Dinv_xps_0pn * WilsonP[n+3] * eps_inv * Dinv_x_0.transpose() * eps * WilsonP[s] ).trace();
            const Complex corr = - tr1 + tr2;

            of << x << " " << y << " "
               << corr.real() << " " << corr.imag() << " "
               << std::endl;
          }}
      }}
  }

  {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){

      int dum1,dum2;
      int sign1 = cshift(dum1, dum2, 0, 0, n);

      std::vector<M2> Dinv_n_0pn;
      if(n==0) Dinv_n_0pn = Dinv_n_A;
      else if(n==1) Dinv_n_0pn = Dinv_n_B;
      else if(n==2) Dinv_n_0pn = Dinv_n_C;
      else assert(false);

      std::ofstream of( dir_data+description+"K"+std::to_string(n)+"E.dat",
                        std::ios::out | std::ios::trunc);
      if(!of) assert(false);
      of << std::scientific << std::setprecision(15);

      for(int y=0; y<Ly; y++){
        for(int x=0; x<Lx; x++){
          if(!is_site(x,y)) continue;
          const int ch = mod(x-y, 3);
          // if(ch!=1) continue;
          // // assert(ch==1);
          if(ch!=0) continue;

          const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
          const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

          const Complex tr1 = 0.5 * ( Dinv_x_0pn * WilsonP[n+3] * eps_inv * Dinv_x_0.transpose() * eps ).trace();
          const Complex tr2 = -0.5 * ( Dinv_x_0pn * WilsonP[n+3] * eps_inv * Dinv_x_0.transpose() * eps ).trace();
          const Complex corr = - tr1 + tr2;

          of << x << " " << y << " "
             << corr.real() << " " << corr.imag() << " "
             << std::endl;
        }}
    }
  }

  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){
      // for(int s=0; s<3; s++){
      for(const int s0 : {0,1,2}){

        int dum1,dum2;
        int sign1 = cshift(dum1, dum2, 0, 0, n);

        std::vector<M2> Dinv_n_0pn;
        if(n==0) Dinv_n_0pn = Dinv_n_A;
        else if(n==1) Dinv_n_0pn = Dinv_n_B;
        else if(n==2) Dinv_n_0pn = Dinv_n_C;
        else assert(false);

        std::ofstream of( dir_data+description+"K"+std::to_string(n)+"E"+std::to_string(s0)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        for(int y=0; y<Ly; y++){
          for(int x=0; x<Lx; x++){
            int s=s0;
            if(!is_site(x,y)) continue;
            // if(!is_link(x,y,s)) continue;

            // if(!is_link(x,y,s)) s+=3;
            // if(!is_link(x,y,s)) continue;
            const int ch = mod(x-y, 3);

            // if(ch!=1) continue;
            // assert(ch==1);
            if(ch!=0) continue;

            const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
            const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

            int xps, yps;
            int sign2 = cshift(xps, yps, x, y, s);
            const M2& Dinv_xps_0 = sign2 * Dinv_n_0[idx(xps,yps)];
            const M2& Dinv_xps_0pn = sign1 * sign2 * Dinv_n_0pn[idx(xps,yps)];

            const Complex tr1 = 0.5 * ( Dinv_xps_0pn * WilsonP[n+3] * eps_inv * Dinv_xps_0.transpose() * eps ).trace();
            const Complex tr2 = -0.5 * ( Dinv_xps_0pn * WilsonP[n+3] * eps_inv * Dinv_xps_0.transpose() * eps ).trace();
            const Complex corr = - tr1 + tr2;

            of << x << " " << y << " "
               << corr.real() << " " << corr.imag() << " "
               << std::endl;
          }}
      }}
  }

  {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    // for(int s=0; s<3; s++){
    for(const int s0 : {0,1,2}){

      std::ofstream of( dir_data+description+"EK"+std::to_string(s0)+".dat",
                        std::ios::out | std::ios::trunc);
      if(!of) assert(false);
      of << std::scientific << std::setprecision(15);

      for(int y=0; y<Ly; y++){
        for(int x=0; x<Lx; x++){
          int s = s0;
          if(!is_site(x,y)) continue;
          // if(!is_link(x,y,s)) continue;

          // if(!is_link(x,y,s)) s+=3;
          // if(!is_link(x,y,s)) continue;
          const int ch = mod(x-y, 3);

          // if(ch!=1) continue;
          // assert(ch==1);
          if(ch!=0) continue;

          const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];

          int xps, yps;
          int sign2 = cshift(xps, yps, x, y, s);
          const M2& Dinv_xps_0 = sign2 * Dinv_n_0[idx(xps,yps)];

          const Complex tr1 = 0.5 * ( Dinv_x_0 * eps_inv * Dinv_xps_0.transpose() * eps * WilsonP[(s+3)%6] ).trace();
          const Complex tr2 = -0.5 * ( Dinv_xps_0 * eps_inv * Dinv_x_0.transpose() * eps * WilsonP[s] ).trace();
          const Complex corr = - tr1 + tr2;

          of << x << " " << y << " "
             << corr.real() << " " << corr.imag() << " "
             << std::endl;
        }}
    }
  }


  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){
      // for(int s=0; s<3; s++){
      for(const int s0 : {0,1,2}){

        int dum1,dum2;
        int sign1 = cshift(dum1, dum2, 0, 0, n);

        std::vector<M2> Dinv_n_0pn;
        if(n==0) Dinv_n_0pn = Dinv_n_A;
        else if(n==1) Dinv_n_0pn = Dinv_n_B;
        else if(n==2) Dinv_n_0pn = Dinv_n_C;
        else assert(false);

        std::ofstream of( dir_data+description+"E"+std::to_string(n)+"K"+std::to_string(s0)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        for(int y=0; y<Ly; y++){
          for(int x=0; x<Lx; x++){
            int s = s0;
            // if(!is_link(x,y,s)) continue;

            // if(!is_link(x,y,s)) s+=3;
            // if(!is_link(x,y,s)) continue;
            // const int ch = mod(x-y, 3);

            if(!is_site(x,y)) continue;
            const int ch = mod(x-y, 3);
            if(ch!=0) continue;

            // if(ch!=1) continue;
            // assert(ch==1);
            // if(ch==0) continue;

            const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

            int xps, yps;
            int sign2 = cshift(xps, yps, x, y, s);
            const M2& Dinv_xps_0pn = sign1 * sign2 * Dinv_n_0pn[idx(xps,yps)];

            const Complex tr1 = 0.5 * ( Dinv_x_0pn * eps_inv * Dinv_xps_0pn.transpose() * eps * WilsonP[(s+3)%6] ).trace();
            const Complex tr2 = -0.5 * ( Dinv_xps_0pn * eps_inv * Dinv_x_0pn.transpose() * eps * WilsonP[s] ).trace();
            const Complex corr = - tr1 + tr2;

            of << x << " " << y << " "
               << corr.real() << " " << corr.imag() << " "
               << std::endl;
          }}
      }}
  }

  {
    std::ofstream of( dir_data+description+"EE.dat",
                      std::ios::out | std::ios::trunc);
    if(!of) assert(false);
    of << std::scientific << std::setprecision(15);

    for(int y=0; y<Ly; y++){
      for(int x=0; x<Lx; x++){
        if(!is_site(x,y)) continue;
        const int ch = mod(x-y, 3);
        if(ch!=0) continue;

        // if(ch!=1) continue;
        // // assert(ch==1);
        // if(ch==0) continue;

        const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
        // const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

        const Complex tr1 = 0.25 * ( Dinv_x_0 * eps_inv * Dinv_x_0.transpose() * eps ).trace();
        const Complex tr2 = -0.25 * ( Dinv_x_0 * eps_inv * Dinv_x_0.transpose() * eps ).trace();
        const Complex corr = - tr1 + tr2;

        of << x << " " << y << " "
           << corr.real() << " " << corr.imag() << " "
           << std::endl;
      }}
  }

  {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){

      int dum1,dum2;
      int sign1 = cshift(dum1, dum2, 0, 0, n);

      std::vector<M2> Dinv_n_0pn;
      if(n==0) Dinv_n_0pn = Dinv_n_A;
      else if(n==1) Dinv_n_0pn = Dinv_n_B;
      else if(n==2) Dinv_n_0pn = Dinv_n_C;
      else assert(false);

      std::ofstream of( dir_data+description+"E"+std::to_string(n)+"E.dat",
                        std::ios::out | std::ios::trunc);
      if(!of) assert(false);
      of << std::scientific << std::setprecision(15);

      for(int y=0; y<Ly; y++){
        for(int x=0; x<Lx; x++){
          if(!is_site(x,y)) continue;
          const int ch = mod(x-y, 3);
          if(ch!=0) continue;

          // if(ch!=1) continue;
          // // assert(ch==1);
          // if(ch==0) continue;

          // const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];
          const M2& Dinv_x_0pn = sign1 * Dinv_n_0pn[idx(x,y)];

          const Complex tr1 = 0.25 * ( Dinv_x_0pn * eps_inv * Dinv_x_0pn.transpose() * eps ).trace();
          const Complex tr2 = -0.25 * ( Dinv_x_0pn * eps_inv * Dinv_x_0pn.transpose() * eps ).trace();
          const Complex corr = - tr1 + tr2;

          of << x << " " << y << " "
             << corr.real() << " " << corr.imag() << " "
             << std::endl;
        }}
    }

    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
      // for(int s=0; s<3; s++){
      for(const int s0 : {0,1,2}){

        std::ofstream of( dir_data+description+"EE"+std::to_string(s0)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        for(int y=0; y<Ly; y++){
          for(int x=0; x<Lx; x++){
            int s = s0;
            if(!is_site(x,y)) continue;
            // if(!is_link(x,y,s)) continue;

            // if(!is_link(x,y,s)) s+=3;
            // if(!is_link(x,y,s)) continue;
            const int ch = mod(x-y, 3);
            if(ch!=0) continue;

            // if(ch!=1) continue;
            // assert(ch==1);
            // if(ch==0) continue;

            // const M2& Dinv_x_0 = Dinv_n_0[idx(x,y)];

            int xps, yps;
            int sign2 = cshift(xps, yps, x, y, s);
            const M2& Dinv_xps_0 = sign2 * Dinv_n_0[idx(xps,yps)];

            const Complex tr1 = 0.25 * ( Dinv_xps_0 * eps_inv * Dinv_xps_0.transpose() * eps ).trace();
            const Complex tr2 = -0.25 * ( Dinv_xps_0 * eps_inv * Dinv_xps_0.transpose() * eps ).trace();
            const Complex corr = - tr1 + tr2;

            of << x << " " << y << " "
               << corr.real() << " " << corr.imag() << " "
               << std::endl;
          }}
      }
    }

  }


  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(nparallel)
#endif
    for(int n=0; n<THREE; n++){
      // for(int s=0; s<3; s++){
      for(const int s0 : {0,1,2}){

        int dum1,dum2;
        int sign1 = cshift(dum1, dum2, 0, 0, n);

        std::vector<M2> Dinv_n_0pn;
        if(n==0) Dinv_n_0pn = Dinv_n_A;
        else if(n==1) Dinv_n_0pn = Dinv_n_B;
        else if(n==2) Dinv_n_0pn = Dinv_n_C;
        else assert(false);

        std::ofstream of( dir_data+description+"E"+std::to_string(n)+"E"+std::to_string(s0)+".dat",
                          std::ios::out | std::ios::trunc);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);

        for(int y=0; y<Ly; y++){
          for(int x=0; x<Lx; x++){
            int s = s0;
            if(!is_site(x,y)) continue;
            // if(!is_link(x,y,s)) continue;

            // if(!is_link(x,y,s)) s+=3;
            // if(!is_link(x,y,s)) continue;
            const int ch = mod(x-y, 3);
            if(ch!=0) continue;

            // if(ch!=1) continue;
            // assert(ch==1);
            // if(ch==0) continue;

            int xps, yps;
            int sign2 = cshift(xps, yps, x, y, s);
            const M2& Dinv_xps_0pn = sign1 * sign2 * Dinv_n_0pn[idx(xps,yps)];

            const Complex tr1 = 0.25 * ( Dinv_xps_0pn * eps_inv * Dinv_xps_0pn.transpose() * eps ).trace();
            const Complex tr2 = -0.25 * ( Dinv_xps_0pn * eps_inv * Dinv_xps_0pn.transpose() * eps ).trace();
            const Complex corr = - tr1 + tr2;

            of << x << " " << y << " "
               << corr.real() << " " << corr.imag() << " "
               << std::endl;
          }}
      }}
  }



  return 0;
}

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>

#include "typedefs_cuda.hpp"
#include "constants.hpp"
#include "header_cuda.hpp"
#include "header_cusolver.hpp"


// using Idx = long long int;
// using Idx = size_t;


int compare (const void * a, const void * b)
{
  return ( *(Idx*)a - *(Idx*)b );
}

int main(int argc, char **argv){
  std::cout << std::scientific << std::setprecision(15) << std::endl;

  int nu=3;

  if (argc>1){
    for (int i = 0; i < argc; i++) {
      nu = atoi(argv[1]);
      printf("%s\n", argv[i]);
    }
  }
  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);

  int device;
  cudacheck(cudaGetDeviceCount(&device));
  cudaDeviceProp device_prop[device];
  cudaGetDeviceProperties(&device_prop[0], 0);
  std::cout << "dev = " << device_prop[0].name << std::endl;
  cudacheck(cudaSetDevice(0));// "TITAN V"
  std::cout << "(GPU device is set.)" << std::endl;


  // -----

  std::cout << "calculating D" << std::endl;

  const Idx N2 = N*N;
  const Idx Neff = N*2/3;

  Complex *D_removed, *LU;

  {
    Complex *e, *D;

    // =====

    std::cout << "-- memory alloc" << std::endl;
    D = (Complex*)malloc(N2*CD);
    set2zero(D, N*N);
    e = (Complex*)malloc(N*CD);

    std::cout << "-- setting" << std::endl;
    for(Idx i=0; i<N; i++){
      set2zero(e, N);
      e[i] = cplx(1.0);
      multD_wrapper( D+i*N, e, nu ); // column major
    }

    free( e );

    //=====

    D_removed = (Complex*)malloc(Neff*Neff*CD);
    set2zero(D_removed, Neff*Neff);

    {
      // std::vector<Idx> vacant;
      // Idx vacant[N/3];
      Idx *vacant;

      vacant = (Idx*)malloc(N/3*sizeof(Idx));
      set2zero(vacant, N/3);

      Idx ii=0;
      for(Idx x=0; x<Lx; x++){
        for(Idx y=0; y<Ly; y++){
          const Idx idx1 = 2*idx(x,y);
          if( !is_site(x,y) ) {
            // vacant.push_back( idx1 );
            // vacant.push_back( idx1+1 );
            vacant[ii]=idx1;
            ii++;
            vacant[ii]=idx1+1;
            ii++;
          }
        }}
      // std::sort(vacant.begin(),vacant.end());
      // std::sort(std::begin(vacant), std::end(vacant));
      std::cout << "sorting..." << std::endl;
      qsort( vacant, N/3, sizeof(Idx), compare );

      Idx idx_tot = 0;

      Idx js=0;
      for(Idx j=0; j<N; j++){
        if(j==vacant[js]){
          js++;
          continue;
        }
        Idx is = 0;
        for(Idx i=0; i<N; i++){
          if(i==vacant[is]){
            is++;
            continue;
          }

          D_removed[idx_tot] = D[i*N+j];
          idx_tot++;
        }}

      free( vacant );
    }

    free( D );
  }

  //=======

  std::cout << "starting LU" << std::endl;

  LU = (Complex*)malloc(Neff*Neff*CD);
  set2zero(LU, Neff*Neff);

  {
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgetrf/cusolver_Xgetrf_example.cu
    // https://github.com/mnicely/cusolver_examples/blob/main/lu_decomposition_cusolver.cu


    // =========================================
    // cusolver
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const int64_t m = Neff;
    const int64_t lda = m;

    int info = 0;

    Complex *d_A;
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;  /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */


    /* step 1: create cusolver handle, bind a stream */
    cudacheck( cusolverDnCreate( &cusolverH ) );
    cudacheck( cudaStreamCreate( &stream ) );
    cudacheck( cusolverDnSetStream( cusolverH, stream ) );


    /* Create advanced params */
    cusolverDnParams_t params;
    cudacheck( cusolverDnCreateParams( &params ) );



    /* step 2: copy A to device */
    std::cout << "Memory allocation" << std::endl;
    cudacheck( cudaMalloc( &d_A, Neff*Neff*CD) );
    cudacheck( cudaMalloc( &d_Ipiv, sizeof( int64_t ) * Neff ) );
    cudacheck( cudaMalloc( &d_info, sizeof( int ) ) );

    cudacheck(cudaMemcpy(d_A, D_removed, Neff*Neff*CD, H2D));


    /* step 3: query working space of getrf */
    cudacheck( cusolverDnXgetrf_bufferSize( cusolverH, NULL, Neff, Neff,
                                            CUDA_C_64F, d_A, Neff, CUDA_C_64F,
                                            &workspaceInBytesOnDevice, &workspaceInBytesOnHost ) );


    cudacheck(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
      h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
      if (h_work == nullptr) {
        throw std::runtime_error("Error: h_work not allocated.");
      }
    }


    /* step 4: LU factorization */
    std::cout << "Executing LU" << std::endl;

    cudacheck(cusolverDnXgetrf(cusolverH, params, m, m, CUDA_C_64F,
                               d_A, lda, d_Ipiv, CUDA_C_64F, d_work,
                               workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    cudacheck(cudaMemcpy(&info, d_info, sizeof(int), D2H));
    cudacheck(cudaMemcpy(LU, d_A, Neff*Neff*CD, D2H));

    std::printf("after Xgetrf: info = %d\n", info);
    if (0 > info) {
      std::printf("%d-th parameter is wrong \n", -info);
      exit(1);
    }

    std::vector<double> zeros;
    double log_abs_det_nozero = 0.0;
    // std::complex<double> det = 1.0;
    for(Idx i=0; i<Neff; i++) {
      // assert( imag(LU[i*Neff+i])<1.0e-14 );
      // log_det += std::log( real(LU[i*Neff+i]) );
      std::complex<double> tmp = real(LU[i*Neff+i]) + std::complex<double>(0.0,1.0)*imag(LU[i*Neff+i]);
      if( abs(arg(tmp))>1.0e-13 ) std::cout << "arg>0: " << tmp << std::endl;
      if( abs(tmp)>1.0e-13 ) log_abs_det_nozero += std::log( abs(tmp) );
      else zeros.push_back( std::abs(tmp) );
    }
    std::cout << "nu = " << nu << std::endl
      //<< "log det = " << log_det << std::endl;
              << "log_abs_det_nozero = " << log_abs_det_nozero << std::endl;

    std::cout << "zeros: " << std::endl;
    for(auto elem : zeros) std::cout << elem << " ";
    std::cout << std::endl;


    // ===========================================

    cudacheck( cudaFree( d_A ) );
    cudacheck( cudaFree( d_Ipiv ) );
    cudacheck( cudaFree( d_info ) );
    cudacheck( cudaFree( d_work ) );
    free(h_work);

    cudacheck(cusolverDnDestroyParams(params));
    cudacheck( cusolverDnDestroy( cusolverH ) );
    cudacheck( cudaStreamDestroy( stream ) );
  }


  free( D_removed );
  free( LU );
  cudaDeviceReset();

  return 0;
}


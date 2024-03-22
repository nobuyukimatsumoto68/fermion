#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <fstream>
#include <algorithm>

#include "typedefs_cuda.hpp"
#include "constants.hpp"
#include "header_cuda.hpp"
#include "header_cusolver.hpp"

// ======================================

//  #define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
//      ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1. ) + (n_)) + (2. / 3.) * (m_)) \
//      : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1. ) + (m_)) + (2. / 3.) * (n_)) )
//  #define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
//      ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1. / 6.) * (m_)) \
//      : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1. / 6.) * (n_)) )
// #define FLOPS_ZGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )


using Idx = long int;


int main(){
  std::cout << std::scientific << std::setprecision(15) << std::endl;

  int device;
  cudacheck(cudaGetDeviceCount(&device));
  cudaDeviceProp device_prop[device];
  cudaGetDeviceProperties(&device_prop[0], 0);
  std::cout << "dev = " << device_prop[0].name << std::endl;
  cudacheck(cudaSetDevice(0));// "TITAN V"
  std::cout << "(GPU device is set.)" << std::endl;


  // -----

  Complex *e, *D, *D_removed, *LU;
  e = (Complex*)malloc(N*CD);

  const Idx N2 = N*N;
  const Idx Neff = N*2/3;

  D = (Complex*)malloc(N2*CD);
  set2zero(D, N*N);

  D_removed = (Complex*)malloc(Neff*Neff*CD);
  set2zero(D_removed, Neff*Neff);
  // Complex D_removed[Neff*Neff];

  LU = (Complex*)malloc(Neff*Neff*CD);
  set2zero(LU, Neff*Neff);

  for(Idx i=0; i<N; i++){
    set2zero(e, N);
    e[i] = cplx(1.0);
    multD_wrapper( D+i*N, e ); // column major
  }


  {
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

        D_removed[idx_tot] = D[i*N+j];
        idx_tot++;
      }}
  }


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
    cudacheck(cudaMalloc(&d_A, Neff*Neff*CD));
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

    double log_det = 0.0;
    for(Idx i=0; i<Neff; i++) {
      assert( imag(LU[i*Neff+i])<1.0e-14 );
      log_det += std::log( real(LU[i*Neff+i]) );
    }
    std::cout << "log det = " << log_det << std::endl;




    // //============================================
    // // obsolete






    // // // Start timer
    // // cudaEvent_t startEvent { nullptr };
    // // cudaEvent_t stopEvent { nullptr };
    // // float       elapsed_gpu_ms {};
    // // cudacheck( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    // // cudacheck( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    // std::printf( "Pivot is on : compute P*A = L*U\n" );

    // // std::memcpy(m_A, D_removed, Neff*Neff*CD);
    // // cudaDeviceSynchronize( );
    // std::printf( "Using New Algo\n" );
    // cudacheck( cusolverDnSetAdvOptions( params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0 ) );


    // /* Create advanced params */




    // // memory

    // void *bufferOnDevice { nullptr };
    // void *bufferOnHost { nullptr };


    // std::printf( "\nAllocate device workspace, lwork = %lu\n", workspaceInBytesOnDevice );
    // std::printf( "Allocate host workspace, lwork = %lu\n\n", workspaceInBytesOnHost );

    // // cudacheck( cudaMallocManaged( &bufferOnDevice, workspaceInBytesOnDevice ) );
    // size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    // void *d_work = nullptr;              /* device workspace for getrf */
    // size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    // void *h_work = nullptr;              /* host workspace for getrf */

    // if ( 0 < workspaceInBytesOnHost ) {
    //   cudacheck( cudaMallocManaged( &bufferOnHost, workspaceInBytesOnHost ) );
    //   assert( NULL != bufferOnHost );
    // }

    // // Create advanced params
    // // if ( algo == 0 ) {
    // //   std::printf( "Using New Algo\n" );
    // //   cudacheck( cusolverDnSetAdvOptions( params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0 ) );
    // // } else {
    // //   std::printf( "Using Legacy Algo\n" );
    // //   cudacheck( cusolverDnSetAdvOptions( params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1 ) );
    // // }


    // /* step 4: LU factorization */
    // std::printf( "\nRunning GETRF\n" );
    // // cudacheck( cudaEventRecord( startEvent ) );

    // cudacheck( cusolverDnXgetrf( cusolverH,
    //                              params,
    //                              static_cast<int64_t>( Neff ),
    //                              static_cast<int64_t>( Neff ),
    //                              CUDA_C_64F,
    //                              d_A,
    //                              static_cast<int64_t>( Neff ),
    //                              d_Ipiv,
    //                              CUDA_C_64F,
    //                              bufferOnDevice,
    //                              workspaceInBytesOnDevice,
    //                              bufferOnHost,
    //                              workspaceInBytesOnHost,
    //                              d_info ) );

    // cudacheck(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
    // if (0 < workspaceInBytesOnHost) {
    //   h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
    //   if (h_work == nullptr) {
    //     throw std::runtime_error("Error: h_work not allocated.");
    //   }
    // }

    // // Must be here to retrieve d_info
    // // cudacheck( cudaStreamSynchronize( stream ) );
    // if ( *d_info ) {
    //   throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrf) \n" );
    // }

    // // Stop timer
    // cudacheck( cudaEventRecord( stopEvent ) );
    // cudacheck( cudaEventSynchronize( stopEvent ) );

    // cudacheck( cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) );
    // double avg { elapsed_gpu_ms };
    // double flops { FLOPS_ZGETRF( Neff, Neff ) };
    // double perf { 1e-9 * flops / avg };
    // std::printf( "\nRuntime = %0.2f ms : @ %0.2f GFLOPs\n\n", avg, perf );



    // ===========================================



    // cudacheck( cudaEventDestroy( startEvent ) );
    // cudacheck( cudaEventDestroy( stopEvent ) );

    cudacheck( cudaFree( d_A ) );
    cudacheck( cudaFree( d_Ipiv ) );
    cudacheck( cudaFree( d_info ) );
    cudacheck( cudaFree( d_work ) );
    free(h_work);

    cudacheck(cusolverDnDestroyParams(params));
    cudacheck( cusolverDnDestroy( cusolverH ) );
    cudacheck( cudaStreamDestroy( stream ) );

    // cudacheck( cudaFree( bufferOnDevice ) );
    // cudacheck( cudaFree( bufferOnHost ) );

  }


  free( e );
  free( D );
  free( D_removed );
  free( LU );
  cudaDeviceReset();

  return 0;
}


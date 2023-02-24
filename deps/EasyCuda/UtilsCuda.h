#pragma once

//pytorch
#include <c10/cuda/CUDACachingAllocator.h>

//cuda 
#include <cuda.h>

//eigen
#include <Eigen/Dense>

//c++
#include <iostream>

//my stuff
// #include "surfel_renderer/utils/MiscUtils.h"

//opencv
#include "opencv2/opencv.hpp"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include "loguru/loguru.hpp" //needs to be added after torch.h otherwise loguru stops printing for some reason


// #define CUD_C(stmt) do { 
//     stmt;
//     gpuAssert(#stmt, __FILE__, __LINE__);	
// } while (0)
inline void gpuAssert(const char* stmt, const char* fname, int line)
{
   auto code = cudaGetLastError();
   if (code != cudaSuccess) 
   {
      printf("CUDA error %s, at %s:%i - for %s.\n", cudaGetErrorString(code), fname, line, stmt);
      exit(1);
   }

//    // More careful checking. However, this will affect performance.
//     // Comment away if needed.
//     auto err = cudaDeviceSynchronize();
//     if( cudaSuccess != err )
//     {
//         fprintf( stderr, "cudaCheckError() with sync failed at %s\n",
//                  cudaGetErrorString( err ) );
//         exit( -1 );
//     }


}
// CUD_C Check Macro. Will terminate the program if a error is detected.
#define CUDA_CHECK_ERROR() { gpuAssert("none", __FILE__, __LINE__);	}


// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {

//     // fprintf(stdout, "coda is %s \n", cudaGetErrorString(code));
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

// // CUD_C Check Macro. Will terminate the program if a error is detected.
// #define CUDA_CHECK_ERROR() { gpuErrchk( cudaGetLastError() ); gpuErrchk( cudaDeviceSynchronize() );	}



inline void check_curesult(const CUresult& res, const char* fname, int line){
    const char *err_description;      
    const char *err_name;      
    

    cuGetErrorString(res, &err_description);     
    cuGetErrorName(res, &err_name);     
    if ( res!=CUDA_SUCCESS){
      printf("CUDA error %s: %s, at %s:%i .\n", err_name, err_description, fname, line);
      exit(1);
   }

}
#define CUDA_CHECK_CURESULT(res) { check_curesult(res, __FILE__, __LINE__);	}


// inline std::string cuResult2string(const CUresult& res){
//     // std::string str;
//     // const char *cstr = str.c_str();
//     const char *msg;      
//     // cuGetErrorName(res, &msg );



//     cuGetErrorString(res, &msg);     
//     std::cerr << "\nerror:  failed with error "       
//                 << msg << '\n';                                  

//     std::string str=msg;

//     return str;

// }


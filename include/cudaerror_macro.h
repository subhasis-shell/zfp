#ifndef __CUDAERROR_MACRO__
#define __CUDAERROR_MACRO__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

/* CUDA specific canonical error checking */

#define checkCudaError(ans) { gpuCheck((ans), __FILE__, __LINE__); }

inline void gpuCheck(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

/*
#define checkCudaError(ans) if((ans) != cudaSuccess) \
                            fprintf(stderr, "Cuda Failure in file \
                              %s: %d: '%s'\n", __FILE__, __LINE__, \
                              cudaGetErrorString(cudaGetLastError()))
*/

#endif

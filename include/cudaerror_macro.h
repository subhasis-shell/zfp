#ifndef __CUDAERROR_MACRO__
#define __CUDAERROR_MACRO__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

/* CUDA specific canonical error checking */

#define checkCudaError(ans) if((ans) != cudaSuccess) \
                            fprintf(stderr, "Cuda Failure in file \
                              %s: %d: '%s'\n", __FILE__, __LINE__, \
                              cudaGetErrorString(cudaGetLastError()))
                     
#endif

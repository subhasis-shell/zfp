#ifndef __HIPERROR_MACRO__
#define __HIPERROR_MACRO__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <hip/hip_common.h>

/* HIP specific canonical error checking */

#define checkHipError(ans) { gpuCheck((ans), __FILE__, __LINE__); }

inline void gpuCheck(hipError_t code, const char *file, int line)
{
    if (code != hipSuccess) {
          fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
              exit(code);
                }
}

#endif


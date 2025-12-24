#ifndef IMAGE_FFT_H
#define IMAGE_FFT_H

#include <iostream>
#include <string>
#include <vector>

// CUDA Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // IMAGE_FFT_H
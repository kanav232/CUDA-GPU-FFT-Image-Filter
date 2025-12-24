#ifndef KERNELS_H
#define KERNELS_H

#include <cufft.h>

// Kernel to convert input float image to Complex format (Real=pixel, Imag=0)
__global__ void floatToComplex(const float* input, cufftComplex* output, int width, int height);

// Kernel to apply Low-Pass Filter in Frequency Domain
// Zeros out frequencies outside the specified radius
__global__ void applyLowPassFilter(cufftComplex* data, int width, int height, float radius);

// Kernel to normalize Inverse FFT result and convert back to real float
__global__ void complexToRealNormalized(const cufftComplex* input, float* output, int width, int height);

// Kernel to compute magnitude spectrum for visualization: log(1 + |z|)
__global__ void computeLogMagnitude(const cufftComplex* input, float* output, int width, int height);

#endif // KERNELS_H
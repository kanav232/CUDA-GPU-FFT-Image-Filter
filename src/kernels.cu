#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void floatToComplex(const float *input, cufftComplex *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        output[idx].x = input[idx]; // Real part
        output[idx].y = 0.0f;       // Imaginary part
    }
}

__global__ void applyLowPassFilter(cufftComplex *data, int width, int height, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        // In standard FFT output, (0,0) is the DC component.
        // Frequencies increase towards the center (Nyquist).
        // We need to calculate distance from the "corners" (0,0) accounting for periodicity.

        // Shifted coordinates to center the spectrum logic
        int i = (x > width / 2) ? (x - width) : x;
        int j = (y > height / 2) ? (y - height) : y;

        float dist = sqrtf((float)(i * i + j * j));

        if (dist > radius)
        {
            // High frequency -> Cut off
            data[idx].x = 0.0f;
            data[idx].y = 0.0f;
        }
    }
}

__global__ void complexToRealNormalized(const cufftComplex *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        // cuFFT inverse transform is unnormalized. We must divide by N (total pixels).
        float normFactor = 1.0f / (float)(width * height);

        output[idx] = input[idx].x * normFactor;
    }
}

__global__ void computeLogMagnitude(const cufftComplex *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        // Magnitude = sqrt(re^2 + im^2)
        float mag = cuCabsf(input[idx]);

        // Log scale for better visualization: log(1 + mag)
        output[idx] = logf(1.0f + mag);
    }
}
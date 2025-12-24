# CUDA GPU FFT Image Filter

## Project Overview

This project demonstrates high-performance image processing using CUDA and cuFFT. It implements a frequency-domain Low-Pass Filter (LPF) to blur images. The pipeline converts an image to the frequency domain, removes high-frequency components (edges/noise) using a custom CUDA kernel, and reconstructs the image.

## GPU Implementation Details

1.  **Memory Transfer**: Image data is converted to float and moved from Host (CPU) to Device (GPU).
2.  **FFT**: `cufftExecC2C` performs a Forward Fast Fourier Transform.
3.  **Custom Kernel**: `applyLowPassFilter` operates on the complex frequency data. It calculates the distance of each frequency component from the DC component (0,0) and zeros out values exceeding the specified radius.
4.  **Inverse FFT**: `cufftExecC2C` (Inverse) transforms data back to the spatial domain.
5.  **Normalization**: A custom kernel normalizes the cuFFT output (dividing by $N$) and prepares it for saving.

## Build Instructions

Requirements: Linux, NVIDIA GPU, CUDA Toolkit, `wget`.

1.  Make the run script executable:
    ```bash
    chmod +x run.sh
    ```
2.  Run the script (downloads dependencies, compiles, and runs):
    ```bash
    ./run.sh
    ```

## Output

The program generates three images:

1.  `output_original.png`: The grayscale input.
2.  `output_magnitude.png`: The frequency spectrum (log-scaled) before filtering.
3.  `output_filtered.png`: The resulting blurred image after low-pass filtering.

## Lessons Learned

- **Complex Plane**: cuFFT C2C transforms result in a data layout where the DC component is at index 0. Filtering logic must account for periodicity (negative frequencies appear in the second half of the array).
- **Normalization**: Unlike MATLAB, cuFFT inverse transforms are unnormalized, requiring a manual division by the total number of pixels.
- **Memory Coalescing**: Using 2D CUDA grids (`blockIdx.x`, `blockIdx.y`) maps naturally to image coordinates and ensures efficient memory access patterns.

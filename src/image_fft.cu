#include "image_fft.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cufft.h>

// STB Image implementation (Define only once in the project)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void saveImage(const char *filename, float *data, int width, int height, bool normalize_0_255)
{
    std::vector<unsigned char> output_data(width * height);

    float min_val = 1e9;
    float max_val = -1e9;

    // Find range if normalization is needed
    if (normalize_0_255)
    {
        for (int i = 0; i < width * height; i++)
        {
            if (data[i] < min_val)
                min_val = data[i];
            if (data[i] > max_val)
                max_val = data[i];
        }
    }

    for (int i = 0; i < width * height; ++i)
    {
        float val = data[i];
        if (normalize_0_255)
        {
            // Normalize to 0-255 based on min/max
            val = (val - min_val) / (max_val - min_val) * 255.0f;
        }
        else
        {
            // Assume data is already roughly 0-255, just clamp
            if (val < 0.0f)
                val = 0.0f;
            if (val > 255.0f)
                val = 255.0f;
        }
        output_data[i] = static_cast<unsigned char>(val);
    }

    stbi_write_png(filename, width, height, 1, output_data.data(), width);
    printf("Saved: %s\n", filename);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <input_image> <filter_radius>\n", argv[0]);
        return 1;
    }

    const char *inputFilename = argv[1];
    float filterRadius = atof(argv[2]);

    // 1. Load Image
    int width, height, channels;
    unsigned char *h_img_in_uchar = stbi_load(inputFilename, &width, &height, &channels, 1); // Force grayscale
    if (!h_img_in_uchar)
    {
        printf("Error loading image %s\n", inputFilename);
        return 1;
    }
    printf("Loaded Image: %dx%d\n", width, height);

    size_t num_pixels = width * height;
    size_t bytes_float = num_pixels * sizeof(float);
    size_t bytes_complex = num_pixels * sizeof(cufftComplex);

    // Convert uchar to float for processing
    float *h_img_in_float = (float *)malloc(bytes_float);
    for (size_t i = 0; i < num_pixels; i++)
    {
        h_img_in_float[i] = (float)h_img_in_uchar[i];
    }

    // 2. Allocate Device Memory
    float *d_img_in, *d_img_out, *d_mag_out;
    cufftComplex *d_complex_data;

    gpuErrchk(cudaMalloc((void **)&d_img_in, bytes_float));
    gpuErrchk(cudaMalloc((void **)&d_img_out, bytes_float));
    gpuErrchk(cudaMalloc((void **)&d_mag_out, bytes_float));
    gpuErrchk(cudaMalloc((void **)&d_complex_data, bytes_complex));

    // 3. Copy Host to Device
    gpuErrchk(cudaMemcpy(d_img_in, h_img_in_float, bytes_float, cudaMemcpyHostToDevice));

    // Setup Grid/Block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Setup Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Setup cuFFT Plan
    cufftHandle plan;
    if (cufftPlan2d(&plan, height, width, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT Plan creation failed\n");
        return 1;
    }

    printf("Starting GPU Processing...\n");
    cudaEventRecord(start);

    // --- STEP A: Convert Real to Complex ---
    floatToComplex<<<grid, block>>>(d_img_in, d_complex_data, width, height);
    gpuErrchk(cudaGetLastError());

    // --- STEP B: Forward FFT ---
    if (cufftExecC2C(plan, d_complex_data, d_complex_data, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT Forward exec failed\n");
        return 1;
    }

    // (Optional) Compute Magnitude for visualization before filtering
    computeLogMagnitude<<<grid, block>>>(d_complex_data, d_mag_out, width, height);

    // --- STEP C: Apply Filter Kernel ---
    applyLowPassFilter<<<grid, block>>>(d_complex_data, width, height, filterRadius);
    gpuErrchk(cudaGetLastError());

    // --- STEP D: Inverse FFT ---
    if (cufftExecC2C(plan, d_complex_data, d_complex_data, CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT Inverse exec failed\n");
        return 1;
    }

    // --- STEP E: Normalize and Real Conversion ---
    complexToRealNormalized<<<grid, block>>>(d_complex_data, d_img_out, width, height);
    gpuErrchk(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Processing Time: %.3f ms\n", milliseconds);

    // 4. Copy back to Host
    float *h_img_out = (float *)malloc(bytes_float);
    float *h_mag_out = (float *)malloc(bytes_float);

    gpuErrchk(cudaMemcpy(h_img_out, d_img_out, bytes_float, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_mag_out, d_mag_out, bytes_float, cudaMemcpyDeviceToHost));

    // 5. Save Results
    saveImage("output_original.png", h_img_in_float, width, height, false);
    saveImage("output_magnitude.png", h_mag_out, width, height, true); // Normalize log mag
    saveImage("output_filtered.png", h_img_out, width, height, false);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_mag_out);
    cudaFree(d_complex_data);
    stbi_image_free(h_img_in_uchar);
    free(h_img_in_float);
    free(h_img_out);
    free(h_mag_out);

    return 0;
}
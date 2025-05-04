// dvd.shivam: CUDA kernels and GPU processing functions
#include "imgxform_dvd.hpp"
#include <cuda.h>
#include <iostream>

// GPU constants and kernels
__constant__ int imgxform_gauss5x5_dev[25]; // dvd.shivam: Gaussian kernel in constant memory (same as CPU kernel)

// Kernel to compute per-block partial sums for R, G, B channels
__global__ void imgxform_kernel_sum(const unsigned char *d_data, int totalPixels, unsigned long long *d_partialSumR, unsigned long long *d_partialSumG, unsigned long long *d_partialSumB) { // dvd.shivam
    // Each block will produce one partial sum for each channel
    unsigned long long localSumR = 0;
    unsigned long long localSumG = 0;
    unsigned long long localSumB = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Stride loop to cover all pixels
    for (int i = idx; i < totalPixels; i += blockDim.x * gridDim.x) {
        // Each pixel has 3 bytes (R, G, B)
        int base = 3 * i;
        localSumR += d_data[base + 0];
        localSumG += d_data[base + 1];
        localSumB += d_data[base + 2];
    }
    // Use shared memory reduction within block
    __shared__ unsigned long long s_sumR[256];
    __shared__ unsigned long long s_sumG[256];
    __shared__ unsigned long long s_sumB[256];
    int t = threadIdx.x;
    s_sumR[t] = localSumR;
    s_sumG[t] = localSumG;
    s_sumB[t] = localSumB;
    __syncthreads();
    // Reduce in shared memory
    if (t < 128) {
        s_sumR[t] += s_sumR[t+128];
        s_sumG[t] += s_sumG[t+128];
        s_sumB[t] += s_sumB[t+128];
    }
    __syncthreads();
    if (t < 64) {
        s_sumR[t] += s_sumR[t+64];
        s_sumG[t] += s_sumG[t+64];
        s_sumB[t] += s_sumB[t+64];
    }
    __syncthreads();
    if (t < 32) {
        // unrolling warp
        s_sumR[t] += s_sumR[t+32];
        s_sumG[t] += s_sumG[t+32];
        s_sumB[t] += s_sumB[t+32];
        s_sumR[t] += s_sumR[t+16];
        s_sumG[t] += s_sumG[t+16];
        s_sumB[t] += s_sumB[t+16];
        s_sumR[t] += s_sumR[t+8];
        s_sumG[t] += s_sumG[t+8];
        s_sumB[t] += s_sumB[t+8];
        s_sumR[t] += s_sumR[t+4];
        s_sumG[t] += s_sumG[t+4];
        s_sumB[t] += s_sumB[t+4];
        s_sumR[t] += s_sumR[t+2];
        s_sumG[t] += s_sumG[t+2];
        s_sumB[t] += s_sumB[t+2];
        s_sumR[t] += s_sumR[t+1];
        s_sumG[t] += s_sumG[t+1];
        s_sumB[t] += s_sumB[t+1];
    }
    __syncthreads();
    if (t == 0) {
        // Write block result
        d_partialSumR[blockIdx.x] = s_sumR[0];
        d_partialSumG[blockIdx.x] = s_sumG[0];
        d_partialSumB[blockIdx.x] = s_sumB[0];
    }
}

// Kernel to scale each pixel by given color scale factors (gray-world)
__global__ void imgxform_kernel_apply_scale(unsigned char *d_data, int totalPixels, float scaleR, float scaleG, float scaleB) { // dvd.shivam
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalPixels) {
        int base = 3 * i;
        float r = d_data[base + 0] * scaleR;
        float g = d_data[base + 1] * scaleG;
        float b = d_data[base + 2] * scaleB;
        int ri = __float2int_rn(r);
        int gi = __float2int_rn(g);
        int bi = __float2int_rn(b);
        if (ri > 255) ri = 255; if (ri < 0) ri = 0;
        if (gi > 255) gi = 255; if (gi < 0) gi = 0;
        if (bi > 255) bi = 255; if (bi < 0) bi = 0;
        d_data[base + 0] = static_cast<unsigned char>(ri);
        d_data[base + 1] = static_cast<unsigned char>(gi);
        d_data[base + 2] = static_cast<unsigned char>(bi);
    }
}

// Kernel to compute per-block min and max for each channel
__global__ void imgxform_kernel_minmax(const unsigned char *d_data, int totalPixels, unsigned char *d_partialMinMax) { // dvd.shivam
    // d_partialMinMax size = gridDim.x * 6 (Rmin, Gmin, Bmin, Rmax, Gmax, Bmax for each block)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char localMinR = 255, localMinG = 255, localMinB = 255;
    unsigned char localMaxR = 0, localMaxG = 0, localMaxB = 0;
    for (int i = idx; i < totalPixels; i += blockDim.x * gridDim.x) {
        int base = 3 * i;
        unsigned char r = d_data[base + 0];
        unsigned char g = d_data[base + 1];
        unsigned char b = d_data[base + 2];
        if (r < localMinR) localMinR = r;
        if (g < localMinG) localMinG = g;
        if (b < localMinB) localMinB = b;
        if (r > localMaxR) localMaxR = r;
        if (g > localMaxG) localMaxG = g;
        if (b > localMaxB) localMaxB = b;
    }
    __shared__ unsigned char s_minR[256], s_minG[256], s_minB[256];
    __shared__ unsigned char s_maxR[256], s_maxG[256], s_maxB[256];
    int t = threadIdx.x;
    s_minR[t] = localMinR;
    s_minG[t] = localMinG;
    s_minB[t] = localMinB;
    s_maxR[t] = localMaxR;
    s_maxG[t] = localMaxG;
    s_maxB[t] = localMaxB;
    __syncthreads();
    // Reduction for min and max (parallel reduction)
    if (t < 128) {
        if (s_minR[t+128] < s_minR[t]) s_minR[t] = s_minR[t+128];
        if (s_minG[t+128] < s_minG[t]) s_minG[t] = s_minG[t+128];
        if (s_minB[t+128] < s_minB[t]) s_minB[t] = s_minB[t+128];
        if (s_maxR[t+128] > s_maxR[t]) s_maxR[t] = s_maxR[t+128];
        if (s_maxG[t+128] > s_maxG[t]) s_maxG[t] = s_maxG[t+128];
        if (s_maxB[t+128] > s_maxB[t]) s_maxB[t] = s_maxB[t+128];
    }
    __syncthreads();
    if (t < 64) {
        if (s_minR[t+64] < s_minR[t]) s_minR[t] = s_minR[t+64];
        if (s_minG[t+64] < s_minG[t]) s_minG[t] = s_minG[t+64];
        if (s_minB[t+64] < s_minB[t]) s_minB[t] = s_minB[t+64];
        if (s_maxR[t+64] > s_maxR[t]) s_maxR[t] = s_maxR[t+64];
        if (s_maxG[t+64] > s_maxG[t]) s_maxG[t] = s_maxG[t+64];
        if (s_maxB[t+64] > s_maxB[t]) s_maxB[t] = s_maxB[t+64];
    }
    __syncthreads();
    if (t < 32) {
        // unroll
        if (s_minR[t+32] < s_minR[t]) s_minR[t] = s_minR[t+32];
        if (s_minG[t+32] < s_minG[t]) s_minG[t] = s_minG[t+32];
        if (s_minB[t+32] < s_minB[t]) s_minB[t] = s_minB[t+32];
        if (s_maxR[t+32] > s_maxR[t]) s_maxR[t] = s_maxR[t+32];
        if (s_maxG[t+32] > s_maxG[t]) s_maxG[t] = s_maxG[t+32];
        if (s_maxB[t+32] > s_maxB[t]) s_maxB[t] = s_maxB[t+32];
        if (s_minR[t+16] < s_minR[t]) s_minR[t] = s_minR[t+16];
        if (s_minG[t+16] < s_minG[t]) s_minG[t] = s_minG[t+16];
        if (s_minB[t+16] < s_minB[t]) s_minB[t] = s_minB[t+16];
        if (s_maxR[t+16] > s_maxR[t]) s_maxR[t] = s_maxR[t+16];
        if (s_maxG[t+16] > s_maxG[t]) s_maxG[t] = s_maxG[t+16];
        if (s_maxB[t+16] > s_maxB[t]) s_maxB[t] = s_maxB[t+16];
        if (s_minR[t+8] < s_minR[t]) s_minR[t] = s_minR[t+8];
        if (s_minG[t+8] < s_minG[t]) s_minG[t] = s_minG[t+8];
        if (s_minB[t+8] < s_minB[t]) s_minB[t] = s_minB[t+8];
        if (s_maxR[t+8] > s_maxR[t]) s_maxR[t] = s_maxR[t+8];
        if (s_maxG[t+8] > s_maxG[t]) s_maxG[t] = s_maxG[t+8];
        if (s_maxB[t+8] > s_maxB[t]) s_maxB[t] = s_maxB[t+8];
        if (s_minR[t+4] < s_minR[t]) s_minR[t] = s_minR[t+4];
        if (s_minG[t+4] < s_minG[t]) s_minG[t] = s_minG[t+4];
        if (s_minB[t+4] < s_minB[t]) s_minB[t] = s_minB[t+4];
        if (s_maxR[t+4] > s_maxR[t]) s_maxR[t] = s_maxR[t+4];
        if (s_maxG[t+4] > s_maxG[t]) s_maxG[t] = s_maxG[t+4];
        if (s_maxB[t+4] > s_maxB[t]) s_maxB[t] = s_maxB[t+4];
        if (s_minR[t+2] < s_minR[t]) s_minR[t] = s_minR[t+2];
        if (s_minG[t+2] < s_minG[t]) s_minG[t] = s_minG[t+2];
        if (s_minB[t+2] < s_minB[t]) s_minB[t] = s_minB[t+2];
        if (s_maxR[t+2] > s_maxR[t]) s_maxR[t] = s_maxR[t+2];
        if (s_maxG[t+2] > s_maxG[t]) s_maxG[t] = s_maxG[t+2];
        if (s_maxB[t+2] > s_maxB[t]) s_maxB[t] = s_maxB[t+2];
        if (s_minR[t+1] < s_minR[t]) s_minR[t] = s_minR[t+1];
        if (s_minG[t+1] < s_minG[t]) s_minG[t] = s_minG[t+1];
        if (s_minB[t+1] < s_minB[t]) s_minB[t] = s_minB[t+1];
        if (s_maxR[t+1] > s_maxR[t]) s_maxR[t] = s_maxR[t+1];
        if (s_maxG[t+1] > s_maxG[t]) s_maxG[t] = s_maxG[t+1];
        if (s_maxB[t+1] > s_maxB[t]) s_maxB[t] = s_maxB[t+1];
    }
    __syncthreads();
    if (t == 0) {
        int baseOut = blockIdx.x * 6;
        d_partialMinMax[baseOut + 0] = s_minR[0];
        d_partialMinMax[baseOut + 1] = s_minG[0];
        d_partialMinMax[baseOut + 2] = s_minB[0];
        d_partialMinMax[baseOut + 3] = s_maxR[0];
        d_partialMinMax[baseOut + 4] = s_maxG[0];
        d_partialMinMax[baseOut + 5] = s_maxB[0];
    }
}

// Kernel to apply contrast stretching (min-max normalization)
__global__ void imgxform_kernel_apply_contrast(unsigned char *d_data, int totalPixels, unsigned char minR, unsigned char minG, unsigned char minB, float scaleR, float scaleG, float scaleB) { // dvd.shivam
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalPixels) {
        int base = 3 * i;
        unsigned char r = d_data[base + 0];
        unsigned char g = d_data[base + 1];
        unsigned char b = d_data[base + 2];
        int rr = __float2int_rn((r - minR) * scaleR);
        int gg = __float2int_rn((g - minG) * scaleG);
        int bb = __float2int_rn((b - minB) * scaleB);
        if (rr < 0) rr = 0; if (rr > 255) rr = 255;
        if (gg < 0) gg = 0; if (gg > 255) gg = 255;
        if (bb < 0) bb = 0; if (bb > 255) bb = 255;
        d_data[base + 0] = static_cast<unsigned char>(rr);
        d_data[base + 1] = static_cast<unsigned char>(gg);
        d_data[base + 2] = static_cast<unsigned char>(bb);
    }
}

// Kernel for 5x5 Gaussian blur (with clamp border)
__global__ void imgxform_kernel_gaussian_blur(const unsigned char *d_input, unsigned char *d_output, int width, int height) { // dvd.shivam
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx < totalPixels) {
        int x = idx % width;
        int y = idx / width;
        int kernelSize = 5;
        int radius = 2;
        int sumR = 0, sumG = 0, sumB = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
            int yy = y + ky;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;
            for (int kx = -radius; kx <= radius; ++kx) {
                int xx = x + kx;
                if (xx < 0) xx = 0;
                if (xx >= width) xx = width - 1;
                int w = imgxform_gauss5x5_dev[(ky+radius)*kernelSize + (kx+radius)];
                int base = (yy * width + xx) * 3;
                sumR += w * d_input[base + 0];
                sumG += w * d_input[base + 1];
                sumB += w * d_input[base + 2];
            }
        }
        // Normalize by 256
        sumR >>= 8;
        sumG >>= 8;
        sumB >>= 8;
        if (sumR > 255) sumR = 255; if (sumR < 0) sumR = 0;
        if (sumG > 255) sumG = 255; if (sumG < 0) sumG = 0;
        if (sumB > 255) sumB = 255; if (sumB < 0) sumB = 0;
        int outBase = idx * 3;
        d_output[outBase + 0] = static_cast<unsigned char>(sumR);
        d_output[outBase + 1] = static_cast<unsigned char>(sumG);
        d_output[outBase + 2] = static_cast<unsigned char>(sumB);
    }
}

// Kernel for 3x3 Laplacian sharpening
__global__ void imgxform_kernel_sharpen(const unsigned char *d_input, unsigned char *d_output, int width, int height) { // dvd.shivam
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx < totalPixels) {
        int x = idx % width;
        int y = idx / width;
        int radius = 1;
        int sumR = 0, sumG = 0, sumB = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
            int yy = y + ky;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;
            for (int kx = -radius; kx <= radius; ++kx) {
                int xx = x + kx;
                if (xx < 0) xx = 0;
                if (xx >= width) xx = width - 1;
                int w = (kx == 0 && ky == 0) ? 9 : -1;
                int base = (yy * width + xx) * 3;
                sumR += w * d_input[base + 0];
                sumG += w * d_input[base + 1];
                sumB += w * d_input[base + 2];
            }
        }
        if (sumR < 0) sumR = 0; if (sumR > 255) sumR = 255;
        if (sumG < 0) sumG = 0; if (sumG > 255) sumG = 255;
        if (sumB < 0) sumB = 0; if (sumB > 255) sumB = 255;
        int outBase = idx * 3;
        d_output[outBase + 0] = static_cast<unsigned char>(sumR);
        d_output[outBase + 1] = static_cast<unsigned char>(sumG);
        d_output[outBase + 2] = static_cast<unsigned char>(sumB);
    }
}

// Host wrapper functions for GPU stages
void imgxform_color_balance_gpu(unsigned char *d_data, int width, int height) { // dvd.shivam
    int totalPixels = width * height;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    // Allocate partial sum arrays for each channel
    unsigned long long *d_partialSumR = nullptr, *d_partialSumG = nullptr, *d_partialSumB = nullptr;
    cudaMalloc(&d_partialSumR, blocks * sizeof(unsigned long long));
    cudaMalloc(&d_partialSumG, blocks * sizeof(unsigned long long));
    cudaMalloc(&d_partialSumB, blocks * sizeof(unsigned long long));
    // Launch kernel to compute partial sums
    imgxform_kernel_sum<<<blocks, threads>>>(d_data, totalPixels, d_partialSumR, d_partialSumG, d_partialSumB);
    // Copy partial results to host and accumulate
    std::vector<unsigned long long> h_sumR(blocks), h_sumG(blocks), h_sumB(blocks);
    cudaMemcpy(h_sumR.data(), d_partialSumR, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sumG.data(), d_partialSumG, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sumB.data(), d_partialSumB, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_partialSumR);
    cudaFree(d_partialSumG);
    cudaFree(d_partialSumB);
    unsigned long long sumR = 0, sumG = 0, sumB = 0;
    for (int i = 0; i < blocks; ++i) {
        sumR += h_sumR[i];
        sumG += h_sumG[i];
        sumB += h_sumB[i];
    }
    double avgR = sumR / (double) totalPixels;
    double avgG = sumG / (double) totalPixels;
    double avgB = sumB / (double) totalPixels;
    double avgGray = (avgR + avgG + avgB) / 3.0;
    float scaleR = (avgR > 0.0) ? (float)(avgGray / avgR) : 1.0f;
    float scaleG = (avgG > 0.0) ? (float)(avgGray / avgG) : 1.0f;
    float scaleB = (avgB > 0.0) ? (float)(avgGray / avgB) : 1.0f;
    // Apply scaling on GPU
    imgxform_kernel_apply_scale<<<blocks, threads>>>(d_data, totalPixels, scaleR, scaleG, scaleB);
    cudaDeviceSynchronize();
}

void imgxform_contrast_stretch_gpu(unsigned char *d_data, int width, int height) { // dvd.shivam
    int totalPixels = width * height;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    // Allocate partial min/max array (6 values per block)
    unsigned char *d_partialMinMax = nullptr;
    cudaMalloc(&d_partialMinMax, blocks * 6 * sizeof(unsigned char));
    imgxform_kernel_minmax<<<blocks, threads>>>(d_data, totalPixels, d_partialMinMax);
    std::vector<unsigned char> h_partial(6 * blocks);
    cudaMemcpy(h_partial.data(), d_partialMinMax, 6 * blocks * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_partialMinMax);
    // Reduce partial results on host
    unsigned char minR = 255, minG = 255, minB = 255;
    unsigned char maxR = 0, maxG = 0, maxB = 0;
    for (int b = 0; b < blocks; ++b) {
        unsigned char rmin = h_partial[b*6 + 0];
        unsigned char gmin = h_partial[b*6 + 1];
        unsigned char bmin = h_partial[b*6 + 2];
        unsigned char rmax = h_partial[b*6 + 3];
        unsigned char gmax = h_partial[b*6 + 4];
        unsigned char bmax = h_partial[b*6 + 5];
        if (rmin < minR) minR = rmin;
        if (gmin < minG) minG = gmin;
        if (bmin < minB) minB = bmin;
        if (rmax > maxR) maxR = rmax;
        if (gmax > maxG) maxG = gmax;
        if (bmax > maxB) maxB = bmax;
    }
    float scaleR = (maxR > minR) ? 255.0f / (maxR - minR) : 1.0f;
    float scaleG = (maxG > minG) ? 255.0f / (maxG - minG) : 1.0f;
    float scaleB = (maxB > minB) ? 255.0f / (maxB - minB) : 1.0f;
    imgxform_kernel_apply_contrast<<<blocks, threads>>>(d_data, totalPixels, minR, minG, minB, scaleR, scaleG, scaleB);
    cudaDeviceSynchronize();
}

void imgxform_gaussian_blur_gpu(const unsigned char *d_input, unsigned char *d_output, int width, int height) { // dvd.shivam
    int totalPixels = width * height;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    imgxform_kernel_gaussian_blur<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
}

void imgxform_sharpen_gpu(const unsigned char *d_input, unsigned char *d_output, int width, int height) { // dvd.shivam
    int totalPixels = width * height;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    imgxform_kernel_sharpen<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
}
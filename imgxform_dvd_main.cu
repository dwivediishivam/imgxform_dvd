// dvd.shivam: Main program for imgxform_dvd
#include "imgxform_dvd.hpp"
#include <iostream>
#include <cstring>

// Forward declaration of device constant kernel array
extern __constant__ int imgxform_gauss5x5_dev[25]; // dvd.shivam

int main(int argc, char *argv[]) { // dvd.shivam
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm> [-cpu]" << std::endl;
        return 1;
    }
    const char *inputFile = argv[1];
    const char *outputFile = argv[2];
    bool useCPU = false;
    if (argc >= 4 && std::strcmp(argv[3], "-cpu") == 0) {
        useCPU = true;
    }
    // Load input image
    int width = 0, height = 0;
    std::vector<unsigned char> image;
    if (!imgxform_load_ppm(inputFile, width, height, image)) {
        return 1;
    }
    if (useCPU) {
        // CPU pipeline
        imgxform_color_balance_cpu(image.data(), width, height);
        imgxform_contrast_stretch_cpu(image.data(), width, height);
        std::vector<unsigned char> temp(image.size());
        imgxform_gaussian_blur_cpu(image.data(), temp.data(), width, height);
        // reuse image buffer for sharpen output
        imgxform_sharpen_cpu(temp.data(), image.data(), width, height);
    } else {
        // GPU pipeline
        unsigned char *d_image = nullptr;
        unsigned char *d_temp = nullptr;
        size_t dataSize = image.size() * sizeof(unsigned char);
        // Initialize Gaussian kernel in constant memory
        // Define host Gaussian kernel weights array
        const int h_gauss5x5[25] = {
            1, 4, 6, 4, 1,
            4,16,24,16, 4,
            6,24,36,24, 6,
            4,16,24,16, 4,
            1, 4, 6, 4, 1
        };
        cudaMemcpyToSymbol(imgxform_gauss5x5_dev, h_gauss5x5, sizeof(h_gauss5x5));
        // Allocate device memory
        if (cudaMalloc(&d_image, dataSize) != cudaSuccess) {
            std::cerr << "Error: cudaMalloc failed for image buffer." << std::endl;
            return 1;
        }
        if (cudaMalloc(&d_temp, dataSize) != cudaSuccess) {
            std::cerr << "Error: cudaMalloc failed for temp buffer." << std::endl;
            cudaFree(d_image);
            return 1;
        }
        // Copy input image to device
        cudaMemcpy(d_image, image.data(), dataSize, cudaMemcpyHostToDevice);
        // Execute GPU stages
        imgxform_color_balance_gpu(d_image, width, height);
        imgxform_contrast_stretch_gpu(d_image, width, height);
        imgxform_gaussian_blur_gpu(d_image, d_temp, width, height);
        imgxform_sharpen_gpu(d_temp, d_image, width, height);
        // Copy result back to host
        cudaMemcpy(image.data(), d_image, dataSize, cudaMemcpyDeviceToHost);
        // Free device memory
        cudaFree(d_temp);
        cudaFree(d_image);
    }
    // Save output image
    if (!imgxform_save_ppm(outputFile, width, height, image.data())) {
        return 1;
    }
    return 0;
}
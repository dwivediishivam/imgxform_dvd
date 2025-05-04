// dvd.shivam: Unified declarations for the imgxform_dvd project
#ifndef IMGXFORM_DVD_HPP_
#define IMGXFORM_DVD_HPP_

#include <vector>

// ─────────────────────────────────────────────────────────────
//  PPM I/O  (imgxform_dvd_ppmio.cpp)
// ─────────────────────────────────────────────────────────────
bool imgxform_load_ppm(const char *filename,
                       int &width,
                       int &height,
                       std::vector<unsigned char> &data); // dvd.shivam

bool imgxform_save_ppm(const char *filename,
                       int width,
                       int height,
                       const unsigned char *data); // dvd.shivam

// ─────────────────────────────────────────────────────────────
//  CPU implementations  (imgxform_dvd_utils.cpp)
// ─────────────────────────────────────────────────────────────
void imgxform_color_balance_cpu(unsigned char *data,
                                int width,
                                int height); // dvd.shivam

void imgxform_contrast_stretch_cpu(unsigned char *data,
                                   int width,
                                   int height); // dvd.shivam

void imgxform_gaussian_blur_cpu(const unsigned char *input,
                                unsigned char *output,
                                int width,
                                int height); // dvd.shivam

void imgxform_sharpen_cpu(const unsigned char *input,
                          unsigned char *output,
                          int width,
                          int height); // dvd.shivam

// ─────────────────────────────────────────────────────────────
//  GPU pipeline wrappers  (imgxform_dvd_kernels.cu)
// ─────────────────────────────────────────────────────────────
void imgxform_color_balance_gpu(unsigned char *d_data,
                                int width,
                                int height); // dvd.shivam

void imgxform_contrast_stretch_gpu(unsigned char *d_data,
                                   int width,
                                   int height); // dvd.shivam

void imgxform_gaussian_blur_gpu(const unsigned char *d_input,
                                unsigned char *d_output,
                                int width,
                                int height); // dvd.shivam

void imgxform_sharpen_gpu(const unsigned char *d_input,
                          unsigned char *d_output,
                          int width,
                          int height); // dvd.shivam

#endif // IMGXFORM_DVD_HPP_

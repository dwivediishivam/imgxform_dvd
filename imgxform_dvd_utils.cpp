// dvd.shivam: CPU image processing implementations
#include "imgxform_dvd.hpp"
#include <algorithm>
#include <cmath>

// Color balance (Gray-World) on CPU: scale each channel to equalize their averages.
void imgxform_color_balance_cpu(unsigned char *data, int width, int height)
{ // dvd.shivam
    long long sumR = 0, sumG = 0, sumB = 0;
    int totalPixels = width * height;
    for (int i = 0; i < totalPixels; ++i)
    {
        sumR += data[3 * i + 0];
        sumG += data[3 * i + 1];
        sumB += data[3 * i + 2];
    }
    // Compute average of each channel
    double avgR = sumR / (double)totalPixels;
    double avgG = sumG / (double)totalPixels;
    double avgB = sumB / (double)totalPixels;
    double avgGray = (avgR + avgG + avgB) / 3.0;
    // Compute scale factors for each channel
    double scaleR = avgGray / (avgR > 0.0 ? avgR : 1.0);
    double scaleG = avgGray / (avgG > 0.0 ? avgG : 1.0);
    double scaleB = avgGray / (avgB > 0.0 ? avgB : 1.0);
    // Apply scaling to each pixel
    for (int i = 0; i < totalPixels; ++i)
    {
        int r = static_cast<int>(std::round(data[3 * i + 0] * scaleR));
        int g = static_cast<int>(std::round(data[3 * i + 1] * scaleG));
        int b = static_cast<int>(std::round(data[3 * i + 2] * scaleB));
        if (r > 255)
            r = 255;
        if (g > 255)
            g = 255;
        if (b > 255)
            b = 255;
        data[3 * i + 0] = static_cast<unsigned char>(r < 0 ? 0 : r);
        data[3 * i + 1] = static_cast<unsigned char>(g < 0 ? 0 : g);
        data[3 * i + 2] = static_cast<unsigned char>(b < 0 ? 0 : b);
    }
}

// Contrast stretching (min-max normalization) on CPU
void imgxform_contrast_stretch_cpu(unsigned char *data, int width, int height)
{ // dvd.shivam
    int totalPixels = width * height;
    int minR = 255, minG = 255, minB = 255;
    int maxR = 0, maxG = 0, maxB = 0;
    for (int i = 0; i < totalPixels; ++i)
    {
        int r = data[3 * i + 0];
        int g = data[3 * i + 1];
        int b = data[3 * i + 2];
        if (r < minR)
            minR = r;
        if (g < minG)
            minG = g;
        if (b < minB)
            minB = b;
        if (r > maxR)
            maxR = r;
        if (g > maxG)
            maxG = g;
        if (b > maxB)
            maxB = b;
    }
    // Compute scale factors for each channel
    double scaleR = (maxR > minR) ? 255.0 / (maxR - minR) : 1.0;
    double scaleG = (maxG > minG) ? 255.0 / (maxG - minG) : 1.0;
    double scaleB = (maxB > minB) ? 255.0 / (maxB - minB) : 1.0;
    // Apply scaling to each pixel
    for (int i = 0; i < totalPixels; ++i)
    {
        int r = data[3 * i + 0];
        int g = data[3 * i + 1];
        int b = data[3 * i + 2];
        int rr = static_cast<int>(std::round((r - minR) * scaleR));
        int gg = static_cast<int>(std::round((g - minG) * scaleG));
        int bb = static_cast<int>(std::round((b - minB) * scaleB));
        if (rr < 0)
            rr = 0;
        if (rr > 255)
            rr = 255;
        if (gg < 0)
            gg = 0;
        if (gg > 255)
            gg = 255;
        if (bb < 0)
            bb = 0;
        if (bb > 255)
            bb = 255;
        data[3 * i + 0] = static_cast<unsigned char>(rr);
        data[3 * i + 1] = static_cast<unsigned char>(gg);
        data[3 * i + 2] = static_cast<unsigned char>(bb);
    }
}

// 5x5 Gaussian blur filter kernel (integer weights sum to 256)
static const int imgxform_gauss5x5[25] = { // dvd.shivam
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1};

// Gaussian blur (5x5) on CPU
void imgxform_gaussian_blur_cpu(const unsigned char *input, unsigned char *output, int width, int height)
{ // dvd.shivam
    int kernelRadius = 2;
    int kernelSize = 5;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            long sumR = 0, sumG = 0, sumB = 0;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
            {
                int yy = y + ky;
                if (yy < 0)
                    yy = 0;
                if (yy >= height)
                    yy = height - 1;
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                {
                    int xx = x + kx;
                    if (xx < 0)
                        xx = 0;
                    if (xx >= width)
                        xx = width - 1;
                    int w = imgxform_gauss5x5[(ky + kernelRadius) * kernelSize + (kx + kernelRadius)];
                    const unsigned char *pix = &input[(yy * width + xx) * 3];
                    sumR += w * pix[0];
                    sumG += w * pix[1];
                    sumB += w * pix[2];
                }
            }
            // Normalize by 256 (sum of weights)
            sumR >>= 8;
            sumG >>= 8;
            sumB >>= 8;
            if (sumR > 255)
                sumR = 255;
            if (sumG > 255)
                sumG = 255;
            if (sumB > 255)
                sumB = 255;
            output[(y * width + x) * 3 + 0] = static_cast<unsigned char>((sumR < 0) ? 0 : sumR);
            output[(y * width + x) * 3 + 1] = static_cast<unsigned char>((sumG < 0) ? 0 : sumG);
            output[(y * width + x) * 3 + 2] = static_cast<unsigned char>((sumB < 0) ? 0 : sumB);
        }
    }
}

// Sharpen using Laplacian kernel (3x3) on CPU
void imgxform_sharpen_cpu(const unsigned char *input, unsigned char *output, int width, int height)
{ // dvd.shivam
    // 3x3 Laplacian sharpening kernel: [-1 -1 -1; -1 9 -1; -1 -1 -1]
    int kernelRadius = 1;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int sumR = 0, sumG = 0, sumB = 0;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
            {
                int yy = y + ky;
                if (yy < 0)
                    yy = 0;
                if (yy >= height)
                    yy = height - 1;
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                {
                    int xx = x + kx;
                    if (xx < 0)
                        xx = 0;
                    if (xx >= width)
                        xx = width - 1;
                    int w;
                    if (kx == 0 && ky == 0)
                    {
                        w = 9;
                    }
                    else
                    {
                        w = -1;
                    }
                    const unsigned char *pix = &input[(yy * width + xx) * 3];
                    sumR += w * pix[0];
                    sumG += w * pix[1];
                    sumB += w * pix[2];
                }
            }
            // Clamp results to [0,255]
            if (sumR < 0)
                sumR = 0;
            if (sumR > 255)
                sumR = 255;
            if (sumG < 0)
                sumG = 0;
            if (sumG > 255)
                sumG = 255;
            if (sumB < 0)
                sumB = 0;
            if (sumB > 255)
                sumB = 255;
            output[(y * width + x) * 3 + 0] = static_cast<unsigned char>(sumR);
            output[(y * width + x) * 3 + 1] = static_cast<unsigned char>(sumG);
            output[(y * width + x) * 3 + 2] = static_cast<unsigned char>(sumB);
        }
    }
}

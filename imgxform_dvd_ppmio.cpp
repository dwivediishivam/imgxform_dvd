// dvd.shivam: PPM image input/output
#include "imgxform_dvd.hpp"
#include <iostream>
#include <fstream>
#include <cctype>
#include <string>

// Load a binary PPM (P6) image file into a vector. Returns true if successful.
bool imgxform_load_ppm(const char *filename, int &width, int &height, std::vector<unsigned char> &data)
{ // dvd.shivam
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: could not open input file " << filename << std::endl;
        return false;
    }
    std::string format;
    file >> format;
    if (format != "P6")
    {
        std::cerr << "Error: unsupported PPM format (must be P6)" << std::endl;
        return false;
    }
    // Read width, height
    file >> width >> height;
    if (!file.good())
    {
        std::cerr << "Error: failed to read image dimensions" << std::endl;
        return false;
    }
    int maxval;
    file >> maxval;
    if (!file.good())
    {
        std::cerr << "Error: failed to read maxval" << std::endl;
        return false;
    }
    // Consume the newline character after maxval
    file.get();
    if (maxval != 255)
    {
        std::cerr << "Warning: maxval != 255, this program assumes 8-bit components." << std::endl;
    }
    // Allocate data vector
    data.resize(width * height * 3);
    // Read binary pixel data
    file.read(reinterpret_cast<char *>(data.data()), data.size());
    if (!file)
    {
        std::cerr << "Error: file ended early or read error." << std::endl;
        return false;
    }
    return true;
}

// Save an image (3-channel) to a binary PPM (P6) file.
bool imgxform_save_ppm(const char *filename, int width, int height, const unsigned char *data)
{ // dvd.shivam
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: could not open output file " << filename << std::endl;
        return false;
    }
    file << "P6\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(data), width * height * 3);
    if (!file)
    {
        std::cerr << "Error: failed to write image data to " << filename << std::endl;
        return false;
    }
    return true;
}

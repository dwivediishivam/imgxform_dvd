# imgxform_dvd – GPU‑Accelerated Image‑Enhancement Toolkit

> **Author :** Shivam Dwivedi – _dvd.shivam_
> 
> **Course :** Coursera – GPU Specialization (Johns Hopkins University)

----------

## 1 Project Overview

**imgxform_dvd** is a fully‑CUDA, end‑to‑end image‑enhancement pipeline that runs **entirely inside the Coursera GPU Lab**.  
The tool takes an 8‑bit binary PPM (P6) image, applies four GPU‑accelerated transformations, and writes an enhanced PPM output:

1.  **Gray‑World Colour Balance** → removes colour casts.
    
2.  **Contrast Stretching** → rescales intensity to utilise the full 0‑255 range.
    
3.  **Gaussian Denoise** (5 × 5) → smooths sensor noise.
    
4.  **Laplacian Sharpen** (3 × 3) → restores edge crispness.
    

The CUDA path relies on **NVIDIA Performance Primitives (NPP)** and two lightweight custom kernels. A complete **CPU fallback** (single‑threaded C++) is provided for comparison (`-cpu` flag).

----------

## 2 Key Features

| Feature               | GPU | CPU | Notes                                                  |
|-----------------------|-----|-----|---------------------------------------------------------|
| Colour‑balance        | ✔   | ✔   | Gray‑world algorithm (per‑channel scaling)             |
| Contrast stretch      | ✔   | ✔   | Min‑max normalisation (clamp 0–255)                    |
| Gaussian blur 5×5     | ✔   | ✔   | Integer kernel, weights sum 256                        |
| Laplacian sharpen 3×3 | ✔   | ✔   | Unsharp filter: centre = 9; neighbours = −1            |     |
| Performance log       | ✔   | –   | Prints stage timings in ms (GPU)                       

----------

## 3 Repository Layout

```
imgxform_dvd/
├── Makefile                     # single‑command build
├── imgxform_dvd.hpp             # public API + shared declarations
├── imgxform_dvd_main.cu         # argument parsing + pipeline driver
├── imgxform_dvd_kernels.cu      # CUDA kernels + GPU wrappers
├── imgxform_dvd_utils.cpp       # CPU reference implementations
├── imgxform_dvd_ppmio.cpp       # minimal PPM reader/writer
├── data/
│   ├── input.ppm         # tiny 3‑pixel demo
│   └── output.ppm        # expected 3‑pixel result
└── convert_image_ppm.txt        # one‑liner helper to convert any image → PPM

```

> **convert_image_ppm.txt** contains a single shell command that works inside the Coursera Lab.
> 
> Run it from the terminal to prepare arbitrary formats (PNG/JPEG/WebP…) for **imgxform_dvd**.

----------

## 4 Prerequisites

-   Coursera GPU Lab (CUDA 11 +, NPP linked by default).
    
-   `make` / `nvcc` available in `$PATH` (already true in the lab VM).
    
-   **No** additional libraries or sudo installs required.
    

----------

## 5 Build Instructions

```bash
# 1. Clone / upload the repo into the lab workspace
$ ls
Makefile  imgxform_dvd_main.cu  ...

# 2. Compile (≈ 15 s)
$ make
nvcc  -std=c++11 -O2  -o imgxform_dvd.out *.cu *.cpp -lnppif -lnppc -lnppisu

# 3. Verify
$ ./imgxform_dvd.out --help   # prints usage banner

```

The build produces a single executable: **`imgxform_dvd.out`**.

----------

## 6 Usage

### 6.1 GPU Path (default)

```bash
# Enhance an image on GPU
$ ./imgxform_dvd.out data/sample_input.ppm enhanced.ppm

```


### 6.2 CPU Reference

```bash
# Slow but useful for testing numeric equivalence
$ ./imgxform_dvd.out data/sample_input.ppm cpu_out.ppm -cpu

```

### 6.3 Convert JPEG/PNG → PPM

```bash
# Any image format → binary PPM (P6)
$ bash convert_image_ppm.txt input.png out.ppm

```

## 7 Extending the Project

-   Swap the 5×5 Gaussian for an NPP separable convolution to gain extra FPS.
    
-   Add histogram equalisation (`nppiEqualizeHist`) as a fifth stage.
    
-   Batch mode: iterate over a directory and process every `.ppm` in parallel.
    
-   Integrate cuSOLVER for compressed sensing de‑noise experiments.
    

----------

## 8 License & Originality

All code and documentation © 2025 Shivam Dwivedi (dvd.shivam).  
Created specifically for the Coursera GPU Specialisation capstone project.

_Happy hacking & enjoy lightning‑fast photo fixes!!_

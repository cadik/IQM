# Image Quality Metrics
This library provides Vulkan implementations of SSIM, FSIM, FLIP, PSNR
image quality metrics.

## Prerequisites
- C++ 20
- Vulkan 1.2+
- VkFFT (only for FSIM)

## Binary Usage
Main executable is `IQM`: 
```bash
IQM --method METHOD --input INPUT --ref REF [--output OUTPUT]
```

### Arguments:
- `--method <METHOD>` : selects method to compute, one of SSIM, FSIM, FLIP, PSNR
- `--input <INPUT>` : path to tested image
- `--ref <REF>` : path to reference image
- `--output <OUTPUT>` : path to output image, optional
- `-v, --verbose` : enables more detailed output
- `-c, --colorize `: colorize final output
- `-h, --help` : prints help

### Method specific arguments:
#### PSNR:
- `--psnr-variant <VAR>` : One of `rgb`, `luma` or `yuv`
#### FLIP:
- `--flip-width <WIDTH>` : Width of display in meters
- `--flip-res <RES>` : Resolution of display in pixels
- `--flip-distance <DISTANCE>` : Distance to display in meters

## Library Usage
TODO

## Implemented methods
### PSNR
### SSIM
### FSIM
### FLIP

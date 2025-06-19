# Image Quality Metrics
This library provides Vulkan implementations of SSIM, FSIM, FLIP, PSNR, LPIPS
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
- `--method <METHOD>` : selects method to compute, one of SSIM, FSIM, FLIP, PSNR, LPIPS
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
Example library usage can be found in `/bin/shared/wrappers` folder for each implemented method.
Simplified it can look like this:

```c++
IQM::FLIP flip(device); // init method with `vk::raii::Device`

// populate input structure with Vulkan objects and arguments to method
auto flipInput = IQM::FLIPInput {
    .args = flipArgs,
    ....
};

commandBuffer->begin();
flip.computeMetric(flipInput);
commandBuffer->end();
```

## Compilation
- needs properly configured Vulkan SDK (Tested on Ubuntu 25.04), or packages from system (Arch Linux)
  - in case of bad setup, link errors or missing includes will appear
- for FSIM, git submodule with `VkFFT` must be fetched
- before C++ compilation compile shaders by `./compile_shaders.sh`
- after compilation copy `lpips.dat` next to executable
/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "args.h"
#include "file_matcher.h"
#include "vulkan_instance.h"

#if COMPILE_SSIM
#include "../shared/wrappers/ssim.h"
#endif

#if COMPILE_SVD
#include "../shared/wrappers/svd.h"
#endif

#if COMPILE_FSIM
#include "../shared/wrappers/fsim.h"
#endif

#if COMPILE_FLIP
#include "../shared/wrappers/flip.h"
#endif

#if COMPILE_PSNR
#include "../shared/wrappers/psnr.h"
#endif

#if COMPILE_LPIPS
#include "../shared/wrappers/lpips.h"
#endif

void printHelp() {
    std::cout << "IQM - Application for computing image quality metrics.\n"
    << "Usage: IQM --method METHOD --input INPUT --ref REF [--output OUTPUT]\n\n"
    << "Arguments:\n"
    << "    --method <METHOD> : selects method to compute, one of SSIM, FSIM, FLIP, PSNR\n"
    << "    --input <INPUT>   : path to tested image\n"
    << "    --ref <REF>       : path to reference image\n"
    << "    --output <OUTPUT> : path to output image, optional\n\n"
    << "    -v, --verbose     : enables more detailed output\n"
    << "    -c, --colorize    : colorize final output\n"
    << "    -h, --help        : prints help\n\n"
    << "Method specific arguments:\n"
    << "PSNR:\n"
    << "    --psnr-variant <VAR> : One of `rgb`, `luma` or `yuv`\n"
    << "FLIP:\n"
    << "    --flip-width <WIDTH>       : Width of display in meters\n"
    << "    --flip-res <RES>           : Resolution of display in pixels\n"
    << "    --flip-distance <DISTANCE> : Distance to display in meters\n"
    << std::endl;
}

int main(const int argc, const char **argv) {
    std::optional<IQM::Bin::Args> args;

    try {
        args = IQM::Bin::Args(argc, argv);
    } catch (std::exception& e) {
        std::cout << "Error parsing arguments: " << e.what() << std::endl;
        printHelp();
        return -1;
    }

    if (args->printHelp) {
        printHelp();
        return 0;
    }

    if (args->verbose) {
        std::cout << "Selected method: " << IQM::method_name(args->method) << std::endl;
    }

    IQM::Bin::FileMatcher matcher;
    const auto matches = matcher.match(args.value());

    const IQM::Bin::VulkanInstance vulkan;

    if (args->verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    try {
        switch (args->method) {
            case IQM::Method::SSIM:
#ifdef COMPILE_SSIM
                IQM::Bin::ssim_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("SSIM support is not compiled");
#endif
                break;
            case IQM::Method::CW_SSIM_CPU:
                throw std::runtime_error("CW-SSIM is not implemented");
            case IQM::Method::SVD:
#ifdef COMPILE_SVD
                IQM::Bin::svd_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("M-SVD support is not compiled");
#endif
                break;
            case IQM::Method::FSIM:
#ifdef COMPILE_FSIM
                fsim_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("FSIM support is not compiled");
#endif
                break;
            case IQM::Method::FLIP:
#ifdef COMPILE_FLIP
                flip_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("FLIP support is not compiled");
#endif
                break;
            case IQM::Method::PSNR:
#ifdef COMPILE_PSNR
                psnr_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("PSNR support is not compiled");
#endif
            break;
            case IQM::Method::LPIPS:
#ifdef COMPILE_LPIPS
                lpips_run(args.value(), vulkan, matches);
#else
                throw std::runtime_error("LPIPS support is not compiled");
#endif
            break;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}

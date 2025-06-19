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
#include "../shared/io.h"

#include <GLFW/glfw3.h>
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

void ErrorCallback(int, const char* err_str) {
    std::cout << "GLFW Error: " << err_str << std::endl;
}

void printHelp() {
    std::cout << "IQM-profile - Application for profiling methods in IQM.\n"
    << "Usage: IQM-profile --method METHOD --input INPUT --ref REF [--iterations I]\n\n"
    << "Arguments:\n"
    << "    --method <METHOD>    : selects method to compute, one of SSIM, FSIM, FLIP, PSNR, LPIPS\n"
    << "    --input <INPUT>      : path to tested image\n"
    << "    --ref <REF>          : path to reference image\n"
    << "    -i, --iterations <I> : number of iterations to compute, unlimited if not set\n"
    << "    -v, --verbose        : enables more detailed output\n"
    << "    -c, --colorize       : colorize final output\n"
    << "    -h, --help           : prints help\n\n"
    << "Method specific arguments:\n"
    << "FLIP:\n"
    << "    --flip-width <WIDTH>       : Width of display in meters\n"
    << "    --flip-res <RES>           : Resolution of display in pixels\n"
    << "    --flip-distance <DISTANCE> : Distance to display in meters\n"
    << std::endl;
}

int main(int argc, const char **argv) {
    std::optional<IQM::ProfileArgs> args;

    try {
        args = IQM::ProfileArgs(argc, argv);
    } catch (std::exception& e) {
        std::cout << "Error parsing arguments: " << e.what() << std::endl;
        printHelp();
        return -1;
    }

    if (args->verbose) {
        std::cout << "Selected method: " << IQM::method_name(args->method) << std::endl;
    }

    if (!glfwInit()) {
        return -1;
    }

    glfwSetErrorCallback(ErrorCallback);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "IQM Profile", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    {
        IQM::Profile::VulkanInstance instance(window);

        if (args->verbose) {
            std::cout << "Selected device: "<< instance.selectedDevice << std::endl;
        }

        glfwShowWindow(window);

        const auto input = IQM::Bin::load_image(args->inputPath);
        const auto reference = IQM::Bin::load_image(args->refPath);

        if (input.width != reference.width || input.height != reference.height) {
            throw std::runtime_error("Compared images must have the same size");
        }

        unsigned frameIndex = 0;

#ifdef COMPILE_SSIM
        IQM::SSIM ssim(*instance.device());
#endif
#ifdef COMPILE_SVD
        IQM::SVD svd(*instance.device());
#endif
#ifdef COMPILE_FSIM
        IQM::FSIM fsim(*instance.device());
#endif
#ifdef COMPILE_FLIP
        IQM::FLIP flip(*instance.device());
#endif
#ifdef COMPILE_PSNR
        IQM::PSNR psnr(*instance.device());
#endif
#ifdef COMPILE_LPIPS
        IQM::LPIPS lpips(*instance.device());
        auto modelData = IQM::Bin::load_model("lpips.dat");
#endif

        std::vector<std::chrono::microseconds> times;

        while (!glfwWindowShouldClose(window)) {
            try {
                auto index = instance.acquire();

                auto start = std::chrono::high_resolution_clock::now();

                switch (args->method) {
                    case IQM::Method::SSIM:
#ifdef COMPILE_SSIM
                        IQM::Bin::ssim_run_single(args.value(), instance, ssim, input, reference);
#else
                        throw std::runtime_error("SSIM support is not compiled");
#endif
                    break;
                    case IQM::Method::CW_SSIM_CPU:
                        throw std::runtime_error("CW-SSIM is not implemented");
                    case IQM::Method::SVD:
#ifdef COMPILE_SVD
                        IQM::Bin::svd_run_single(args.value(), instance, svd, input, reference);
#else
                        throw std::runtime_error("M-SVD support is not compiled");
#endif
                    break;
                    case IQM::Method::FSIM:
#ifdef COMPILE_FSIM
                        IQM::Bin::fsim_run_single(args.value(), instance, fsim, input, reference);
#else
                        throw std::runtime_error("FSIM support is not compiled");
#endif
                    break;
                    case IQM::Method::FLIP:
#ifdef COMPILE_FLIP
                        IQM::Bin::flip_run_single(args.value(), instance, flip, input, reference);
#else
                        throw std::runtime_error("FLIP support is not compiled");
#endif
                    break;
                    case IQM::Method::PSNR:
#ifdef COMPILE_PSNR
                        IQM::Bin::psnr_run_single(args.value(), instance, psnr, input, reference);
#else
                        throw std::runtime_error("PSNR support is not compiled");
#endif
                    break;
                    case IQM::Method::LPIPS:
#ifdef COMPILE_LPIPS
                        IQM::Bin::lpips_run_single(args.value(), instance, lpips, input, reference, modelData);
#else
                        throw std::runtime_error("LPIPS support is not compiled");
#endif
                    break;
                }

                auto end = std::chrono::high_resolution_clock::now();

                times.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));

                instance.present(index);
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                exit(-1);
            }

            glfwPollEvents();

            if (args->iterations.has_value()) {
                if (args->iterations.value() <= frameIndex) {
                    break;
                }
            }

            frameIndex += 1;
        }
        instance._device.waitIdle();

        std::sort(times.begin(), times.end());
        auto median = times[times.size() / 2];

        std::cout << "Median run time: " << median << std::endl;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include "args.h"
#include "gpu/base/vulkan_runtime.h"
#include "debug_utils.h"
#include "input_image.h"

#include <GLFW/glfw3.h>

#if COMPILE_SSIM
#include <ssim.h>
#endif

#if COMPILE_SVD
#include <svd.h>
#endif

#if COMPILE_FSIM
#include <fsim.h>
#endif

#if COMPILE_FLIP
#include <flip.h>
#endif

InputImage load_image(const std::string &filename) {
    // force all images to always open in RGBA format to prevent issues with separate RGB and RGBA loading
    int x, y, channels;
    unsigned char* data = stbi_load(filename.c_str(), &x, &y, &channels, 4);
    if (data == nullptr) {
        const auto err = stbi_failure_reason();
        const auto msg = std::string("Failed to load image '" + filename + "', reason: " + err);
        throw std::runtime_error(msg);
    }

    std::vector<unsigned char> dataVec(x * y * 4);
    memcpy(dataVec.data(), data, x * y * 4 * sizeof(char));

    stbi_image_free(data);

    return InputImage{
        .width = x,
        .height = y,
        .data = std::move(dataVec)
    };
}

void ssim(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
#if COMPILE_SSIM
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    IQM::GPU::SSIM ssim(vulkan);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = ssim.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "MSSIM: " << result.mssim << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("SSIM support was not compiled");
#endif
}

void cw_ssim_ref(const IQM::Args& args) {
    /*const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    auto method = IQM::CPU::CW_SSIM_Ref();

    const auto start = std::chrono::high_resolution_clock::now();
    auto out = method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    if (args.verbose) {
        auto execTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << execTime << std::endl;
    }

    return out;*/
}

void svd(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    IQM::GPU::SVD svd(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = svd.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "M-SVD: " << result.msvd << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
}

void fsim(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
#ifdef COMPILE_FSIM
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    IQM::GPU::FSIM fsim(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = fsim.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "FSIM: " << result.fsim << std::endl << "FSIMc: " << result.fsimc << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FSIM support was not compiled");
#endif
}

void flip(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
#ifdef COMPILE_FLIP
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    IQM::GPU::FLIP flip(vulkan);

    auto flip_args = IQM::GPU::FLIPArguments{};
    if (args.options.contains("FLIP_WIDTH")) {
        flip_args.monitor_width = std::stof(args.options.at("FLIP_WIDTH"));
    }
    if (args.options.contains("FLIP_RES")) {
        flip_args.monitor_resolution_x = std::stof(args.options.at("FLIP_RES"));
    }
    if (args.options.contains("FLIP_DISTANCE")) {
        flip_args.monitor_distance = std::stof(args.options.at("FLIP_DISTANCE"));
    }

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl
        << "FLIP monitor resolution: "<< flip_args.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flip_args.monitor_distance << std::endl
        << "FLIP monitor width: "<< flip_args.monitor_width << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = flip.computeMetric(vulkan, input, reference, flip_args);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FLIP support was not compiled");
#endif
}

void ErrorCallback(int, const char* err_str) {
    std::cout << "GLFW Error: " << err_str << std::endl;
}

int main(int argc, const char **argv) {
    auto args = IQM::Args(argc, argv);
    if (args.verbose) {
        std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;
    }

    cv::Mat out;
    const auto outPath = args.outputPath.value_or("out.png");

    if (!glfwInit()) {
        return -1;
    }

    glfwSetErrorCallback(ErrorCallback);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Hello World", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    IQM::GPU::VulkanRuntime vulkan;
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(*vulkan._instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    vulkan.createSwapchain(surface);
    glfwShowWindow(window);

    while (!glfwWindowShouldClose(window)) {
        try {
            auto index = vulkan.acquire();

            switch (args.method) {
                case IQM::Method::SSIM:
                    ssim(args, vulkan);
                break;
                case IQM::Method::CW_SSIM_CPU:
                break;
                case IQM::Method::SVD:
                    svd(args, vulkan);
                break;
                case IQM::Method::FSIM:
                    fsim(args, vulkan);
                case IQM::Method::FLIP:
                    flip(args, vulkan);
                break;
            }

            vulkan.present(index);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            exit(-1);
        }

        glfwPollEvents();
    }

    vulkan._device.waitIdle();
    vulkan.~VulkanRuntime();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

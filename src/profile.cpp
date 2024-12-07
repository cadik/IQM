/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "args.h"
#include "gpu/ssim.h"
#include "gpu/base/vulkan_runtime.h"
#include "debug_utils.h"
#include "cpu/cw_ssim_ref.h"
#include "gpu/svd.h"

#include <GLFW/glfw3.h>

#if COMPILE_FSIM
#include <fsim.h>
#endif

cv::Mat ssim(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
    const cv::Mat image = imread(args.inputPath, cv::ImreadModes::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat imageAlpha;
    cv::Mat refAlpha;

    // convert to correct format for Vulkan
    cvtColor(image, imageAlpha, cv::COLOR_BGR2RGBA);
    cvtColor(ref, refAlpha, cv::COLOR_BGR2RGBA);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    IQM::GPU::SSIM ssim(vulkan);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = ssim.computeMetric(vulkan, imageAlpha, refAlpha);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "MSSIM: " << result.mssim << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    cv::Mat outEightBit = cv::Mat(image.rows, image.cols, CV_8UC1);
    outEightBit = result.image * 255.0;

    return outEightBit;
}

cv::Mat cw_ssim_ref(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    auto method = IQM::CPU::CW_SSIM_Ref();

    const auto start = std::chrono::high_resolution_clock::now();
    auto out = method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    if (args.verbose) {
        auto execTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << execTime << std::endl;
    }

    return out;
}

cv::Mat svd(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
    const cv::Mat image = imread(args.inputPath, cv::ImreadModes::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat imageAlpha;
    cv::Mat refAlpha;

    // convert to correct format for Vulkan
    cvtColor(image, imageAlpha, cv::COLOR_BGR2RGBA);
    cvtColor(ref, refAlpha, cv::COLOR_BGR2RGBA);

    IQM::GPU::SVD svd(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = svd.computeMetric(vulkan, imageAlpha, refAlpha);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "M-SVD: " << result.msvd << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    return result.image;
}

cv::Mat fsim(const IQM::Args& args, const IQM::GPU::VulkanRuntime &vulkan) {
#ifdef COMPILE_FSIM
    const cv::Mat image = imread(args.inputPath, cv::ImreadModes::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat imageAlpha;
    cv::Mat refAlpha;

    // convert to correct format for Vulkan
    cvtColor(image, imageAlpha, cv::COLOR_BGR2RGBA);
    cvtColor(ref, refAlpha, cv::COLOR_BGR2RGBA);

    IQM::GPU::FSIM fsim(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = fsim.computeMetric(vulkan, imageAlpha, refAlpha);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "FSIM: " << result.fsim << std::endl << "FSIMc: " << result.fsimc << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    return result.image;
#else
    throw std::runtime_error("FSIM support was not compiled");
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
                    cw_ssim_ref(args);
                break;
                case IQM::Method::SVD:
                    svd(args, vulkan);
                break;
                case IQM::Method::FSIM:
                    fsim(args, vulkan);
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

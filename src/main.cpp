#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "args.h"
#include "cpu/ssim_ref.h"
#include "gpu/ssim.h"
#include "gpu/base/vulkan_runtime.h"
#include "debug_utils.h"
#include "cpu/cw_ssim_ref.h"
#include "gpu/svd.h"

cv::Mat ssim_ref(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    auto method = IQM::CPU::SSIM_Reference();

    const auto start = std::chrono::high_resolution_clock::now();
    auto out = method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    std::cout << "MSSIM: " << IQM::CPU::SSIM_Reference::computeMSSIM(out) << std::endl;

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << execTime << std::endl;

    return out;
}

cv::Mat ssim(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::ImreadModes::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat imageAlpha;
    cv::Mat refAlpha;

    // convert to correct format for Vulkan
    cvtColor(image, imageAlpha, cv::COLOR_BGR2RGBA);
    cvtColor(ref, refAlpha, cv::COLOR_BGR2RGBA);

    const IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::SSIM ssim(vulkan);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    ssim.prepareImages(vulkan, imageAlpha, refAlpha);
    auto out = ssim.computeMetric(vulkan);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    // convert to correct format for openCV
    cvtColor(out, out, cv::COLOR_RGBA2BGRA);

    std::cout << "MSSIM: " << IQM::CPU::SSIM_Reference::computeMSSIM(out) << std::endl;

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << execTime << std::endl;

    return out;
}

cv::Mat cw_ssim_ref(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    auto method = IQM::CPU::CW_SSIM_Ref();

    const auto start = std::chrono::high_resolution_clock::now();
    auto out = method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << execTime << std::endl;

    return out;
}

cv::Mat svd(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::ImreadModes::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat imageAlpha;
    cv::Mat refAlpha;

    // convert to correct format for Vulkan
    cvtColor(image, imageAlpha, cv::COLOR_BGR2RGBA);
    cvtColor(ref, refAlpha, cv::COLOR_BGR2RGBA);

    const IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::SVD svd(vulkan);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto out = svd.computeMetric(vulkan, imageAlpha, refAlpha);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << execTime << std::endl;

    return out;
}

int main(int argc, const char **argv) {
    auto args = IQM::Args(argc, argv);
    std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;

    cv::Mat out;

    switch (args.method) {
        case IQM::Method::SSIM_CPU:
            out = ssim_ref(args);
            break;
        case IQM::Method::SSIM:
            out = ssim(args);
            break;
        case IQM::Method::CW_SSIM_CPU:
            out = cw_ssim_ref(args);
            break;
        case IQM::Method::SVD:
            out = svd(args);
            break;
    }

    const auto outPath = args.outputPath.value_or("out.png");
    cv::imwrite(outPath, out);
    return 0;
}

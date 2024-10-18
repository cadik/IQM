#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "args.h"
#include "cpu/ssim_ref.h"
#include "gpu/ssim.h"
#include "gpu/base/vulkan_runtime.h"

int main(int argc, char** argv) {
    auto args = IQM::Args(argc, argv);
    std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;

    cv::Mat image = imread(args.input_path, cv::IMREAD_COLOR);
    cv::Mat ref = imread(args.ref_path, cv::IMREAD_COLOR);

    auto method = IQM::CPU::SSIM_Reference();
    auto out = method.computeMetric(image, ref);

    IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::SSIM ssim(vulkan);

    auto start = std::chrono::high_resolution_clock::now();

    ssim.computeMetric(vulkan);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = end - start;
    std::cout << ms_double << std::endl;

    cv::imwrite("out.png", out);
    return 0;
}

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <renderdoc_app.h>

#include "args.h"
#include "cpu/ssim_ref.h"
#include "gpu/ssim.h"
#include "gpu/base/vulkan_runtime.h"

cv::Mat ssim_ref(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    auto method = IQM::CPU::SSIM_Reference();

    const auto start = std::chrono::high_resolution_clock::now();
    auto out = method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << execTime << std::endl;

    return out;
}

cv::Mat ssim(const IQM::Args& args) {
    const cv::Mat image = imread(args.inputPath, cv::IMREAD_COLOR);
    const cv::Mat ref = imread(args.refPath, cv::IMREAD_COLOR);

    IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::SSIM ssim(vulkan);

    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
    if(void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        std::cout << "Renderdoc loaded" << std::endl;
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)(dlsym(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        assert(ret == 1);
    }

    if (rdoc_api) {
        rdoc_api->StartFrameCapture(nullptr, nullptr);
    }

    auto start = std::chrono::high_resolution_clock::now();
    ssim.prepareImages(vulkan, image, ref);
    auto out = ssim.computeMetric(vulkan);
    auto end = std::chrono::high_resolution_clock::now();

    cvtColor(out, out, cv::COLOR_RGBA2BGRA);

    if (rdoc_api) {
        auto res = rdoc_api->EndFrameCapture(nullptr, nullptr);
        std::cout << (res ? "ok" : "nok") << std::endl;
    }

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
    }

    const auto outPath = args.outputPath.value_or("out.png");
    cv::imwrite(outPath, out);
    return 0;
}

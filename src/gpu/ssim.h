#ifndef SSIM_H
#define SSIM_H

#include <opencv2/core/mat.hpp>

#include "img_params.h"
#include "base/vulkan_runtime.h"

namespace IQM::GPU {
    class SSIM {
    public:
        explicit SSIM(const VulkanRuntime &runtime);
        cv::Mat computeMetric(const VulkanRuntime &runtime);
        void prepareImages(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);

        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
        float sigma = 1.5;
    private:
        ImageParameters imageParameters;

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        VulkanImage imageInput;
        VulkanImage imageRef;
        VulkanImage imageOut;
    };
}

#endif //SSIM_H

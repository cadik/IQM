#ifndef SSIM_H
#define SSIM_H
#include <opencv2/core/mat.hpp>

#include "base/vulkan_runtime.h"

struct ImageParameters {
    unsigned width = 0;
    unsigned height = 0;
};

namespace IQM::GPU {
    class SSIM {
    public:
        explicit SSIM(const VulkanRuntime &runtime);
        cv::Mat computeMetric(const VulkanRuntime &runtime);
        void prepareImages(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);

        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
    private:
        ImageParameters imageParameters;

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        vk::raii::DeviceMemory imageInputMemory = VK_NULL_HANDLE;
        vk::raii::DeviceMemory imageRefMemory = VK_NULL_HANDLE;
        vk::raii::DeviceMemory imageOutputMemory = VK_NULL_HANDLE;

        vk::raii::Image imageInput = VK_NULL_HANDLE;
        vk::raii::Image imageRef = VK_NULL_HANDLE;
        vk::raii::Image imageOutput = VK_NULL_HANDLE;

        vk::raii::ImageView imageInputView = VK_NULL_HANDLE;
        vk::raii::ImageView imageRefView = VK_NULL_HANDLE;
        vk::raii::ImageView imageOutputView = VK_NULL_HANDLE;
    };
}

#endif //SSIM_H

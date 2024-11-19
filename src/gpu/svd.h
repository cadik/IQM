#ifndef SVD_H
#define SVD_H

#include <vector>
#include <opencv2/core/mat.hpp>

#include "base/vulkan_runtime.h"
#include "../timestamps.h"

namespace IQM::GPU {
    struct SVDResult {
        cv::Mat image;
        float msvd;
        Timestamps timestamps;
    };

    class SVD {
    public:
        explicit SVD(const VulkanRuntime &runtime);
        SVDResult computeMetric(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);

    private:
        void prepareBuffers(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput);
        void copyToGpu(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput);
        void copyFromGpu(const VulkanRuntime &runtime, size_t sizeOutput);

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        vk::raii::Buffer inputBuffer = VK_NULL_HANDLE;
        vk::raii::Buffer outBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory inputMemory = VK_NULL_HANDLE;
        vk::raii::DeviceMemory outMemory = VK_NULL_HANDLE;

        vk::raii::Buffer stgBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgMemory = VK_NULL_HANDLE;
    };
}

#endif //SVD_H

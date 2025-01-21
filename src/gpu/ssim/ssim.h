/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef SSIM_H
#define SSIM_H

#include <vector>

#include "../../input_image.h"
#include "../img_params.h"
#include "../base/vulkan_runtime.h"
#include "../../timestamps.h"

namespace IQM::GPU {
    struct SSIMResult {
        std::vector<float> imageData;
        unsigned int width;
        unsigned int height;
        float mssim;
        Timestamps timestamps;
    };

    class SSIM {
    public:
        explicit SSIM(const VulkanRuntime &runtime);
        SSIMResult computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);
        [[nodiscard]] double computeMSSIM(const float *buffer, unsigned width, unsigned height) const;

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

        vk::raii::ShaderModule kernelLumapack = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutLumapack = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineLumapack = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetLumapack = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernelGaussInput = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutGaussInput = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGaussInput = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGaussInput = VK_NULL_HANDLE;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;

        vk::raii::Fence transferFence = VK_NULL_HANDLE;
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;
        std::shared_ptr<VulkanImage> imageLuma;
        std::shared_ptr<VulkanImage> imageLumaBlurred;
        std::shared_ptr<VulkanImage> imageOut;

        void prepareImages(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);
    };
}

#endif //SSIM_H

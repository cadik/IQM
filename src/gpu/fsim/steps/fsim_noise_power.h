/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_NOISE_POWER_H
#define FSIM_NOISE_POWER_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    class FSIMNoisePower {
    public:
        explicit FSIMNoisePower(const VulkanRuntime &runtime);
        void computeNoisePower(const VulkanRuntime &runtime, const vk::raii::Buffer &filterSums, const vk::raii::Buffer &fftBuffer, int width, int height);

        vk::raii::DeviceMemory noisePowersMemory = VK_NULL_HANDLE;
        vk::raii::Buffer noisePowers = VK_NULL_HANDLE;

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
    private:
        void copyBackToGpu(const VulkanRuntime &runtime, const vk::raii::Buffer &stgBuf);
        void copyFilterToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer &tempBuf, const vk::raii::Buffer &target, int width, int height);
        void copyFilterSumsToCpu(const VulkanRuntime & runtime, const vk::raii::Buffer & buffer, const vk::raii::Buffer & vk_buffer);
    };
}

#endif //FSIM_NOISE_POWER_H

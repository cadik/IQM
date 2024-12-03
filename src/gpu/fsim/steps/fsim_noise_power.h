/*
 * Image Quality Metrics
 * Petr Volf - 2024
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
    private:
        void copyBackToGpu(const VulkanRuntime &runtime, const vk::raii::Buffer &stgBuf);
        void copyFilterToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer &fftBuf, const vk::raii::Buffer &target, int width, int height, int index);
        void copyFilterSumsToCpu(const VulkanRuntime & runtime, const vk::raii::Buffer & buffer, const vk::raii::Buffer & vk_buffer);
    };
}

#endif //FSIM_NOISE_POWER_H

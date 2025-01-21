/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_ESTIMATE_ENERGY_H
#define FSIM_ESTIMATE_ENERGY_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    /**
     * This step takes the presaved filters and computes estimated noise energy
     */
    class FSIMEstimateEnergy {
    public:
        explicit FSIMEstimateEnergy(const VulkanRuntime &runtime);
        void estimateEnergy(const VulkanRuntime &runtime, const vk::raii::Buffer& fftBuf, int width, int height);

        vk::raii::ShaderModule estimateEnergyKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout estimateEnergyLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline estimateEnergyPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout estimateEnergyDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet estimateEnergyDescSet = VK_NULL_HANDLE;

        vk::raii::ShaderModule sumKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;

        std::vector<vk::raii::Buffer> energyBuffers;
        std::vector<vk::raii::DeviceMemory> energyBuffersMemory;
    private:
        void prepareBufferStorage(const VulkanRuntime& runtime, const vk::raii::Buffer &fftBuf, int width, int height);
    };
}

#endif //FSIM_ESTIMATE_ENERGY_H

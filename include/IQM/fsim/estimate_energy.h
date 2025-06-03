/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_ESTIMATE_ENERGY_H
#define FSIM_ESTIMATE_ENERGY_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FSIMInput;
    /**
     * This step takes the presaved filters and computes estimated noise energy
     */
    class FSIMEstimateEnergy {
        friend class FSIM;
        explicit FSIMEstimateEnergy(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void estimateEnergy(const FSIMInput& input, unsigned width, unsigned height);
        void setUpDescriptors(const FSIMInput& input, unsigned width, unsigned height);

        vk::raii::PipelineLayout estimateEnergyLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline estimateEnergyPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout estimateEnergyDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet estimateEnergyDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;
    };
}

#endif //FSIM_ESTIMATE_ENERGY_H

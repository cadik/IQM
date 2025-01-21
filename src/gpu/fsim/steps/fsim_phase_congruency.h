/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_PHASE_CONGRUENCY_H
#define FSIM_PHASE_CONGRUENCY_H
#include <memory>

#include "../../base/vulkan_runtime.h"
#include "../../base/vulkan_image.h"

namespace IQM::GPU {
    class FSIMPhaseCongruency {
    public:
        explicit FSIMPhaseCongruency(const VulkanRuntime &runtime);
        void compute(const VulkanRuntime &runtime, const vk::raii::Buffer &noiseLevels, const std::vector<vk::raii::Buffer> &energyEstimates, const
                     std::vector<std::shared_ptr<VulkanImage>> &filterResInput, const std::vector<std::shared_ptr<VulkanImage>> &
                     filterResRef, int
                     width, int height);

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> pcInput;
        std::shared_ptr<VulkanImage> pcRef;
    private:
        void prepareImageStorage(const VulkanRuntime &runtime, const vk::raii::Buffer &noiseLevels, const std::vector<vk::raii::Buffer> &energyEstimates, const
                                 std::vector<std::shared_ptr<VulkanImage>> &filterRes, int
                                 width, int height);
    };
}

#endif //FSIM_PHASE_CONGRUENCY_H

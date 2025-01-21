/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_SUM_FILTER_RESPONSES_H
#define FSIM_SUM_FILTER_RESPONSES_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    /**
     * This steps takes the inverse FFT images and computes total energy and amplitude per orientation.
     */
    class FSIMSumFilterResponses {
    public:
        explicit FSIMSumFilterResponses(const VulkanRuntime &runtime);
        void computeSums(const VulkanRuntime &runtime, const vk::raii::Buffer& filters, int width, int height);

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        std::vector<std::shared_ptr<VulkanImage>> filterResponsesInput;
        std::vector<std::shared_ptr<VulkanImage>> filterResponsesRef;
    private:
        void prepareImageStorage(const VulkanRuntime &runtime, const vk::raii::Buffer& filters, int width, int height);
    };
}

#endif //FSIM_SUM_FILTER_RESPONSES_H

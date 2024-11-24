/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_LOG_GABOR_H
#define FSIM_LOG_GABOR_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    class FSIMLogGabor {
    public:
        explicit FSIMLogGabor(const VulkanRuntime &runtime, unsigned scales);
        void constructFilter(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height);

        unsigned scales = 4;

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;

        std::vector<vk::raii::DescriptorSet> descSets;
        std::vector<std::shared_ptr<VulkanImage>> imageLogGaborFilters;
    private:
        void prepareImageStorage(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height);
    };
}

#endif //FSIM_LOG_GABOR_H

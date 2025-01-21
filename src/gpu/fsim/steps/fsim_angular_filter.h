/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_ANGULAR_FILTER_H
#define FSIM_ANGULAR_FILTER_H

#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
    class FSIMAngularFilter {
    public:
        explicit FSIMAngularFilter(const VulkanRuntime &runtime);
        void constructFilter(const VulkanRuntime &runtime, int width, int height);

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
        std::vector<std::shared_ptr<VulkanImage>> imageAngularFilters;
    private:
        void prepareImageStorage(const VulkanRuntime &runtime, int width, int height);
    };
}

#endif //FSIM_ANGULAR_FILTER_H

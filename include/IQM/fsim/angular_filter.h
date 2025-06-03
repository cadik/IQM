/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_ANGULAR_FILTER_H
#define FSIM_ANGULAR_FILTER_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FSIMInput;

    class FSIMAngularFilter {
        friend class FSIM;
        explicit FSIMAngularFilter(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput& input) const;
        void constructFilter(const FSIMInput &input, unsigned width, unsigned height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
    };
}

#endif //FSIM_ANGULAR_FILTER_H

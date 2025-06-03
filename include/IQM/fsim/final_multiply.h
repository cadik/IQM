/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_FINAL_MULTIPLY_H
#define FSIM_FINAL_MULTIPLY_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FSIMInput;

    class FSIMFinalMultiply {
        friend class FSIM;
        explicit FSIMFinalMultiply(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput &input, unsigned width, unsigned height);
        void computeMetrics(const FSIMInput &input, unsigned width, unsigned height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;

        void sumImages(const FSIMInput& input, unsigned width, unsigned height);
    };
}

#endif //FSIM_FINAL_MULTIPLY_H

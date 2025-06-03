/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_SUM_FILTER_RESPONSES_H
#define FSIM_SUM_FILTER_RESPONSES_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FSIMInput;

    /**
     * This steps takes the inverse FFT images and computes total energy and amplitude per orientation.
     */
    class FSIMSumFilterResponses {
        friend class FSIM;
        explicit FSIMSumFilterResponses(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput& input, unsigned width, unsigned height);
        void computeSums(const FSIMInput& input, unsigned width, unsigned height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
    };
}

#endif //FSIM_SUM_FILTER_RESPONSES_H

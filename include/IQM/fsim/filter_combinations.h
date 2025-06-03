/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_FILTER_COMBINATIONS_H
#define FSIM_FILTER_COMBINATIONS_H

#include <IQM/base/vulkan_runtime.h>
#include <IQM/fsim/partitions.h>

namespace IQM {
    struct FSIMInput;
    /**
     * This step takes previously created filters and FFT transformed images
     * and prepares massive buffer for batched inverse FFT done in next step.
     *
     * It also computes noise levels of select filters needed later.
     *
     * The buffer is laid out as such:
     * - gN is log gabor filter of scale N
     * - aN is angular filter of orientation N
     * [ g0 X a0, g1 X a0, ...
     *   g0 X a1, ...
     *   ...
     *   g0 X a0 X img, ...
     *   ...
     *   g0 X a0 X ref, ...
     *   ... ]
     */
    class FSIMFilterCombinations {
        friend class FSIM;
        explicit FSIMFilterCombinations(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput &input, unsigned width, unsigned height);
        void combineFilters(const FSIMInput &input, unsigned width, unsigned height, const FftBufferPartitions& partitions);

        vk::raii::PipelineLayout multPackLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline multPackPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout multPackDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet multPackDescSet = VK_NULL_HANDLE;

        // noise sum part
        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;
    };
}

#endif //FSIM_FILTER_COMBINATIONS_H

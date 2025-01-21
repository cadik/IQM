/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_FILTER_COMBINATIONS_H
#define FSIM_FILTER_COMBINATIONS_H
#include "fsim_angular_filter.h"
#include "fsim_log_gabor.h"
#include "../../base/vulkan_runtime.h"

namespace IQM::GPU {
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
    public:
        explicit FSIMFilterCombinations(const VulkanRuntime &runtime);
        void combineFilters(
            const VulkanRuntime &runtime,
            const FSIMAngularFilter &angulars,
            const FSIMLogGabor &logGabor,
            const vk::raii::Buffer &fftImages,
            int width, int height
        );

        vk::raii::ShaderModule multPackKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout multPacklayout = VK_NULL_HANDLE;
        vk::raii::Pipeline multPackPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout multPackDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet multPackDescSet = VK_NULL_HANDLE;

        vk::raii::Buffer fftBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory fftMemory = VK_NULL_HANDLE;

        // noise sum part
        vk::raii::ShaderModule sumKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;

        vk::raii::Buffer noiseLevels = VK_NULL_HANDLE;
        vk::raii::DeviceMemory noiseLevelsMemory = VK_NULL_HANDLE;
    private:
        void prepareBufferStorage(
            const VulkanRuntime &runtime,
            const FSIMAngularFilter &angulars,
            const FSIMLogGabor &logGabor,
            const vk::raii::Buffer &fftImages,
            int width, int height
        );
    };
}

#endif //FSIM_FILTER_COMBINATIONS_H

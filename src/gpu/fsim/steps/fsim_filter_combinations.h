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
    class FSIMFilterCombinations {
    public:
        explicit FSIMFilterCombinations(const VulkanRuntime &runtime);
        void combineFilters(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars,
                            const FSIMLogGabor &logGabor,
                            int width, int height);


        vk::raii::ShaderModule multPackKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout multPacklayout = VK_NULL_HANDLE;
        vk::raii::Pipeline multPackPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        std::vector<vk::raii::Buffer> buffers;
        std::vector<vk::raii::DeviceMemory> memories;
    private:
        void prepareBufferStorage(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, int width, int height);
    };
}

#endif //FSIM_FILTER_COMBINATIONS_H

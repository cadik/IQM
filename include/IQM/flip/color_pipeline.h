/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_FLIP_COLORPIPELINE_H
#define IQM_FLIP_COLORPIPELINE_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct FLIPInput;

    class FLIPColorPipeline {
    public:
        explicit FLIPColorPipeline(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void prefilter(const FLIPInput& input, float pixels_per_degree);
        void computeErrorMap(const FLIPInput& input);

        void setUpDescriptors(const FLIPInput& input);
    private:
        vk::raii::PipelineLayout csfPrefilterLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline csfPrefilterPipeline = VK_NULL_HANDLE;
        vk::raii::Pipeline csfPrefilterHorizontalPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet csfPrefilterDescSet = VK_NULL_HANDLE;
        vk::raii::DescriptorSet csfPrefilterHorizontalDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout spatialDetectLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline spatialDetectPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet spatialDetectDescSet = VK_NULL_HANDLE;
    };
}

#endif //IQM_FLIP_COLORPIPELINE_H

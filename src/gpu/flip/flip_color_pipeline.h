/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FLIPCOLORPIPELINE_H
#define FLIPCOLORPIPELINE_H

#include "../base/vulkan_runtime.h"
#include "../img_params.h"

namespace IQM::GPU {
    class FLIPColorPipeline {
    public:
        explicit FLIPColorPipeline(const VulkanRuntime &runtime);
        void prepareSpatialFilters(const VulkanRuntime &runtime, int kernel_size, float pixels_per_degree);
        void prefilter(const VulkanRuntime &runtime, ImageParameters params, float pixels_per_degree);
        void computeErrorMap(const VulkanRuntime &runtime, ImageParameters params);

        void prepareStorage(const VulkanRuntime &runtime, int spatial_kernel_size, ImageParameters params);
        void setUpDescriptors(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &inputYcc, const std::shared_ptr<VulkanImage> &refYcc);

        std::shared_ptr<VulkanImage> imageColorError;
    private:
        vk::raii::ShaderModule csfPrefilterKernel = VK_NULL_HANDLE;
        vk::raii::ShaderModule csfPrefilterHorizontalKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout csfPrefilterLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline csfPrefilterPipeline = VK_NULL_HANDLE;
        vk::raii::Pipeline csfPrefilterHorizontalPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout csfPrefilterDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet csfPrefilterDescSet = VK_NULL_HANDLE;
        vk::raii::DescriptorSet csfPrefilterHorizontalDescSet = VK_NULL_HANDLE;

        vk::raii::ShaderModule spatialDetectKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout spatialDetectLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline spatialDetectPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout spatialDetectDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet spatialDetectDescSet = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> csfFilter;

        std::shared_ptr<VulkanImage> inputPrefilterTemp;
        std::shared_ptr<VulkanImage> refPrefilterTemp;

        std::shared_ptr<VulkanImage> inputPrefilter;
        std::shared_ptr<VulkanImage> refPrefilter;
    };
}

#endif //FLIPCOLORPIPELINE_H

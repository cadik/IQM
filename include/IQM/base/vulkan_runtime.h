/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef VULKANRUNTIME_H
#define VULKANRUNTIME_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace IQM::GPU {
    class VulkanRuntime {
    public:
        [[nodiscard]] static vk::raii::ShaderModule createShaderModule(
            const vk::raii::Device &device,
            const std::vector<uint32_t> &spvCode);
        [[nodiscard]] static vk::raii::PipelineLayout createPipelineLayout(
            const vk::raii::Device &device,
            const std::vector<vk::DescriptorSetLayout> &layouts,
            const std::vector<vk::PushConstantRange> &ranges);
        [[nodiscard]] static vk::raii::Pipeline createComputePipeline(
            const vk::raii::Device &device,
            const vk::raii::ShaderModule &shader,
            const vk::raii::PipelineLayout &layout);
        [[nodiscard]] static vk::raii::DescriptorPool createDescPool(
            const vk::raii::Device &device,
            uint32_t maxSets,
            std::vector<vk::DescriptorPoolSize> poolSizes);
        [[nodiscard]] static vk::raii::DescriptorSetLayout createDescLayout(
            const vk::raii::Device &device,
            const std::vector<std::pair<vk::DescriptorType, uint32_t>> &stub);
        static std::vector<vk::PushConstantRange> createPushConstantRange(unsigned size);
        static std::vector<vk::DescriptorImageInfo> createImageInfos(const std::vector<const vk::raii::ImageView *> &images);
        static std::pair<uint32_t, uint32_t> compute2DGroupCounts(const unsigned width, const unsigned height, const unsigned tileSize) {
            auto groupsX = width / tileSize;
            if (width % tileSize != 0) {
                groupsX++;
            }
            auto groupsY = height / tileSize;
            if (height % tileSize != 0) {
                groupsY++;
            }

            return std::make_pair(groupsX, groupsY);
        }

        static uint32_t compute1DGroupCount(const int size, const unsigned blockSize) {
            auto groups = size / blockSize;
            if (size % blockSize != 0) {
                groups++;
            }

            return groups;
        }

        static vk::WriteDescriptorSet createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorImageInfo> &imgInfos);
        static vk::WriteDescriptorSet createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorBufferInfo> &bufInfos);
    };
}

#endif //VULKANRUNTIME_H

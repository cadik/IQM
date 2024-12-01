/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef VULKANRUNTIME_H
#define VULKANRUNTIME_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "vulkan_image.h"

namespace IQM::GPU {
    class VulkanRuntime {
    public:
        VulkanRuntime();
        [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::string& path) const;
        [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout(const std::vector<vk::DescriptorSetLayout> &layouts, const std::vector<vk::PushConstantRange> &ranges) const;
        [[nodiscard]] vk::raii::Pipeline createComputePipeline(const vk::raii::ShaderModule &shader, const vk::raii::PipelineLayout &layout) const;
        [[nodiscard]] std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(unsigned bufferSize, vk::BufferUsageFlags bufferFlags, vk::MemoryPropertyFlags memoryFlags) const;
        [[nodiscard]] VulkanImage createImage(const vk::ImageCreateInfo &imageInfo) const;
        [[nodiscard]] vk::raii::DescriptorSetLayout createDescLayout(const std::vector<vk::DescriptorSetLayoutBinding> &bindings) const;
        void setImageLayout(const std::shared_ptr<vk::raii::CommandBuffer> &cmd_buf, const vk::raii::Image &image, vk::ImageLayout srcLayout, vk::ImageLayout targetLayout) const;
        static void initImages(const std::shared_ptr<vk::raii::CommandBuffer> &cmd_buf, const std::vector<std::shared_ptr<VulkanImage>> &images);
        void nuke() const;
        static std::vector<vk::PushConstantRange> createPushConstantRange(unsigned size);
        static std::vector<vk::DescriptorImageInfo> createImageInfos(const std::vector<std::shared_ptr<VulkanImage>> &images);
        static std::pair<uint32_t, uint32_t> compute2DGroupCounts(const int width, const int height, const int tileSize) {
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
        void waitForFence(const vk::raii::Fence&) const;

        std::string selectedDevice;

        vk::raii::Context _context;
        // assigned VK_NULL_HANDLE to sidestep accidental usage of deleted constructor
        vk::raii::Instance _instance = VK_NULL_HANDLE;
        vk::raii::PhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        vk::raii::Device _device = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::Queue> _queue = VK_NULL_HANDLE;
        uint32_t _queueFamilyIndex;
        std::shared_ptr<vk::raii::Queue> _transferQueue = VK_NULL_HANDLE;
        uint32_t _transferQueueFamilyIndex;
        std::shared_ptr<vk::raii::CommandPool> _commandPool = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandPool> _commandPoolTransfer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> _cmd_buffer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> _cmd_bufferTransfer = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayoutThreeImage = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayoutTwoImage = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayoutOneImage = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayoutBuffer = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout _descLayoutImageBuffer = VK_NULL_HANDLE;
        vk::raii::DescriptorPool _descPool = VK_NULL_HANDLE;

#ifdef PROFILE
        void createSwapchain(vk::SurfaceKHR surface);
        unsigned acquire();
        void present(unsigned index);
        vk::raii::SwapchainKHR swapchain = VK_NULL_HANDLE;
        vk::raii::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        vk::raii::Semaphore renderFinishedSemaphore = VK_NULL_HANDLE;
        vk::raii::Fence swapchainFence = VK_NULL_HANDLE;
#endif
    private:
        void initQueues();
        void initDescriptors();
        static std::vector<const char *> getLayers();
    };
}

#endif //VULKANRUNTIME_H

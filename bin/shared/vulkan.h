/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_VULKAN_H
#define IQM_VULKAN_H

namespace IQM {
    class VulkanInstance {
    public:
        virtual ~VulkanInstance() {}
        virtual const vk::raii::Device* device() const = 0;
        virtual const vk::raii::PhysicalDevice* physicalDevice() const = 0;
        virtual const std::shared_ptr<const vk::raii::CommandPool> cmdPool() const = 0;
        virtual const std::shared_ptr<const vk::raii::CommandBuffer> cmdBuf() const = 0;
        virtual const std::shared_ptr<const vk::raii::CommandBuffer> cmdBufTransfer() const = 0;
        virtual const std::shared_ptr<const vk::raii::Queue> queue() const = 0;
        virtual const std::shared_ptr<const vk::raii::Queue> queueTransfer() const = 0;

        virtual void waitForFence(const vk::raii::Fence &fence) const = 0;
    };
}

#endif //IQM_VULKAN_H

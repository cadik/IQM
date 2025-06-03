/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_PROFILE_VULKAN_INSTANCE_H
#define IQM_PROFILE_VULKAN_INSTANCE_H

#include <string>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_raii.hpp>

#include "../shared/vulkan.h"

namespace IQM::Profile {
    const std::string LAYER_VALIDATION = "VK_LAYER_KHRONOS_validation";

    class VulkanInstance : public IQM::VulkanInstance  {
    public:
        VulkanInstance(GLFWwindow*);

        std::string selectedDevice;

        vk::raii::Context context;
        vk::raii::Instance instance = VK_NULL_HANDLE;
        vk::raii::PhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        vk::raii::Device _device = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::Queue> _queue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex;
        std::shared_ptr<vk::raii::Queue> transferQueue = VK_NULL_HANDLE;
        uint32_t transferQueueFamilyIndex;
        std::shared_ptr<vk::raii::CommandPool> commandPool = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandPool> commandPoolTransfer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> cmd_buffer = VK_NULL_HANDLE;
        std::shared_ptr<vk::raii::CommandBuffer> cmd_bufferTransfer = VK_NULL_HANDLE;

        vk::raii::SurfaceKHR surface = VK_NULL_HANDLE;
        vk::raii::SwapchainKHR swapchain = VK_NULL_HANDLE;
        vk::raii::Semaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        vk::raii::Semaphore renderFinishedSemaphore = VK_NULL_HANDLE;
        vk::raii::Fence swapchainFence = VK_NULL_HANDLE;

        const vk::raii::Device* device() const override {return &_device;}
        const vk::raii::PhysicalDevice* physicalDevice() const override {return &_physicalDevice;}
        const std::shared_ptr<const vk::raii::CommandPool> cmdPool() const override {return commandPool;}
        const std::shared_ptr<const vk::raii::CommandBuffer> cmdBuf() const override {return cmd_buffer;}
        const std::shared_ptr<const vk::raii::CommandBuffer> cmdBufTransfer() const override {return cmd_bufferTransfer;}
        const std::shared_ptr<const vk::raii::Queue> queue() const override {return _queue;}
        const std::shared_ptr<const vk::raii::Queue> queueTransfer() const override {return transferQueue;}

        void createSwapchain();
        unsigned acquire();
        void present(unsigned index);
        void waitForFence(const vk::raii::Fence &fence) const override;
    private:
        void initQueues();
        static std::vector<const char *> getLayers();
    };
}

#endif //IQM_PROFILE_VULKAN_INSTANCE_H

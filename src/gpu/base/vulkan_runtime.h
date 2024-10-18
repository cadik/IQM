#ifndef VULKANRUNTIME_H
#define VULKANRUNTIME_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace IQM::GPU {
    class VulkanRuntime {
    public:
        VulkanRuntime();
        [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::string& path) const;
        [[nodiscard]] vk::raii::PipelineLayout createPipelineLayout(const vk::PipelineLayoutCreateInfo &pipelineLayoutCreateInfo) const;
        [[nodiscard]] vk::raii::Pipeline createComputePipeline(const vk::raii::ShaderModule &shader, const vk::raii::PipelineLayout &layout) const;

    //private:
        vk::raii::Context _context;
        // assigned VK_NULL_HANDLE to sidestep accidental usage of deleted constructor
        vk::raii::Instance _instance = VK_NULL_HANDLE;
        vk::raii::Device _device = VK_NULL_HANDLE;
        vk::raii::Queue _queue = VK_NULL_HANDLE;
        vk::raii::CommandPool _commandPool = VK_NULL_HANDLE;
        vk::raii::CommandBuffer _cmd_buffer = VK_NULL_HANDLE;
    };
}

#endif //VULKANRUNTIME_H

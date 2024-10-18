#include "vulkan_runtime.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

const std::string LAYER_VALIDATION = "VK_LAYER_KHRONOS_validation";

IQM::GPU::VulkanRuntime::VulkanRuntime() {
    this->_context = vk::raii::Context{};

    vk::ApplicationInfo appInfo{
        .pApplicationName = "Image Quality Metrics",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    std::vector layers = {
        LAYER_VALIDATION.c_str()
    };

    std::vector extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    const vk::InstanceCreateInfo instanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 1,
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = extensions.data()
    };

    this->_instance = vk::raii::Instance{this->_context, instanceCreateInfo};

    std::optional<vk::raii::PhysicalDevice> physicalDevice;
    uint32_t computeQueueIndex = 0;

    auto devices = _instance.enumeratePhysicalDevices();
    for (const auto& device : devices) {
        auto properties = device.getProperties();

        physicalDevice = device;

        auto queueFamilyProperties = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilyProperties) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
                computeQueueIndex = i;
            }

            i++;
        }

        std::cout << "Selected device: "<< properties.deviceName << std::endl;
        break;
    }

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{
        .queueFamilyIndex = computeQueueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    const vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
    };

    this->_device = vk::raii::Device{physicalDevice.value(), deviceCreateInfo};
    this->_queue = this->_device.getQueue(computeQueueIndex, 0);

    vk::CommandPoolCreateInfo commandPoolCreateInfo{
        .queueFamilyIndex = computeQueueIndex,
    };

    this->_commandPool = vk::raii::CommandPool{this->_device, commandPoolCreateInfo};

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
        .commandPool = this->_commandPool,
        .commandBufferCount = 1,
    };

    this->_cmd_buffer = std::move(vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo}.front());
}

vk::raii::ShaderModule IQM::GPU::VulkanRuntime::createShaderModule(const std::string &path) const {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    const size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<long>(fileSize));

    file.close();

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo{
        .codeSize = static_cast<uint32_t>(fileSize),
        .pCode = reinterpret_cast<uint32_t*>(buffer.data()),
    };

    vk::raii::ShaderModule module{this->_device, shaderModuleCreateInfo};
    return module;
}

vk::raii::PipelineLayout IQM::GPU::VulkanRuntime::createPipelineLayout(const vk::PipelineLayoutCreateInfo &pipelineLayoutCreateInfo) const {
    return vk::raii::PipelineLayout{this->_device, pipelineLayoutCreateInfo};
}

vk::raii::Pipeline IQM::GPU::VulkanRuntime::createComputePipeline(const vk::raii::ShaderModule &shader, const vk::raii::PipelineLayout &layout) const {
    vk::ComputePipelineCreateInfo computePipelineCreateInfo{
        .stage = vk::PipelineShaderStageCreateInfo {
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = shader,
            // all shaders will start in main
            .pName = "main",
        },
        .layout = layout
    };

    return std::move(vk::raii::Pipelines{this->_device, nullptr, computePipelineCreateInfo}.front());
}


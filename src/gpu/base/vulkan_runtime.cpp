#include "vulkan_runtime.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vulkan_image.h"

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
            if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute && queueFamily.queueFlags & vk::QueueFlagBits::eTransfer && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                computeQueueIndex = i;
                break;
            }

            i++;
        }

        std::cout << "Selected device: "<< properties.deviceName << std::endl;
        break;
    }

    this->_physicalDevice = physicalDevice.value();

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

    this->_device = vk::raii::Device{this->_physicalDevice, deviceCreateInfo};
    this->_queue = this->_device.getQueue(computeQueueIndex, 0);

    vk::CommandPoolCreateInfo commandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = computeQueueIndex,
    };

    this->_commandPool = vk::raii::CommandPool{this->_device, commandPoolCreateInfo};

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
        .commandPool = this->_commandPool,
        .commandBufferCount = 1,
    };

    this->_cmd_buffer = std::move(vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo}.front());

    this->_descLayoutThreeImage = std::move(this->createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
    }));

    this->_descLayoutTwoImage = std::move(this->createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }));

    this->_descLayoutBuffer = std::move(this->createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }));


    std::vector poolSizes = {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 8},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 8}
    };

    vk::DescriptorPoolCreateInfo dsCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 8,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    this->_descPool = std::move(vk::raii::DescriptorPool{this->_device, dsCreateInfo});
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

vk::raii::PipelineLayout IQM::GPU::VulkanRuntime::createPipelineLayout(const std::vector<vk::DescriptorSetLayout> &layouts, const std::vector<vk::PushConstantRange> &ranges) const {
    vk::PipelineLayoutCreateInfo layoutInfo = {
        .flags = {},
        .setLayoutCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(ranges.size()),
        .pPushConstantRanges = ranges.data(),
    };

    return vk::raii::PipelineLayout{this->_device, layoutInfo};
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

uint32_t findMemoryType(vk::PhysicalDeviceMemoryProperties const &memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask) {
    auto typeIndex = static_cast<uint32_t>(~0);
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags & requirementsMask) == requirementsMask)) {
            typeIndex = i;
            break;
        }
        typeBits >>= 1;
    }
    assert(typeIndex != static_cast<uint32_t>(~0));
    return typeIndex;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> IQM::GPU::VulkanRuntime::createBuffer(const unsigned bufferSize, const vk::BufferUsageFlags bufferFlags, const vk::MemoryPropertyFlags memoryFlags) const {
    // create now, so it's destroyed before buffer
    vk::raii::DeviceMemory memory{nullptr};

    vk::BufferCreateInfo bufferCreateInfo{
        .size = bufferSize,
        .usage = bufferFlags,
    };

    vk::raii::Buffer buffer{this->_device, bufferCreateInfo};
    auto memReqs = buffer.getMemoryRequirements();
    const auto memType = findMemoryType(
        this->_physicalDevice.getMemoryProperties(),
        memReqs.memoryTypeBits,
        memoryFlags
    );

    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType
    };

    memory = vk::raii::DeviceMemory{this->_device, memoryAllocateInfo};

    return std::make_pair(std::move(buffer), std::move(memory));
}

IQM::GPU::VulkanImage IQM::GPU::VulkanRuntime::createImage(const vk::ImageCreateInfo &imageInfo) const {
    // create now, so it's destroyed before buffer
    vk::raii::DeviceMemory memory{nullptr};

    vk::raii::Image image{this->_device, imageInfo};
    auto memReqs = image.getMemoryRequirements();
    const auto memType = findMemoryType(
        this->_physicalDevice.getMemoryProperties(),
        memReqs.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType
    };

    memory = vk::raii::DeviceMemory{this->_device, memoryAllocateInfo};
    image.bindMemory(memory, 0);

    vk::ImageViewCreateInfo imageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = imageInfo.format,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    return VulkanImage{
        .memory = std::move(memory),
        .image = std::move(image),
        .imageView = vk::raii::ImageView{this->_device, imageViewCreateInfo},
    };
}

void IQM::GPU::VulkanRuntime::setImageLayout(const vk::raii::Image& image, vk::ImageLayout srcLayout, vk::ImageLayout targetLayout) const {
    vk::AccessFlags sourceAccessMask;
    vk::PipelineStageFlags sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::AccessFlags destinationAccessMask;
    vk::PipelineStageFlags destinationStage = vk::PipelineStageFlagBits::eHost;

    vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;

    vk::ImageSubresourceRange imageSubresourceRange(aspectMask, 0, 1, 0, 1);
    vk::ImageMemoryBarrier imageMemoryBarrier{
        .srcAccessMask = sourceAccessMask,
        .dstAccessMask = destinationAccessMask,
        .oldLayout = srcLayout,
        .newLayout = targetLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = imageSubresourceRange
    };
    return this->_cmd_buffer.pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, imageMemoryBarrier);
}

vk::raii::DescriptorSetLayout IQM::GPU::VulkanRuntime::createDescLayout(const std::vector<vk::DescriptorSetLayoutBinding> &bindings) const {
    auto info = vk::DescriptorSetLayoutCreateInfo {
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return vk::raii::DescriptorSetLayout {this->_device, info};
}
/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#include "vulkan_runtime.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vulkan_image.h"

#ifdef PROFILE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

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

    auto layers = getLayers();

#ifdef PROFILE
    std::vector extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
    };

    uint32_t extensionCount;
    glfwGetRequiredInstanceExtensions(&extensionCount);
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    for (uint32_t i = 0; i < extensionCount; i++) {
        extensions.push_back(glfwExtensions[i]);
    }

#else
    std::vector extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
#endif

    const vk::InstanceCreateInfo instanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    this->_instance = vk::raii::Instance{this->_context, instanceCreateInfo};

    this->initQueues();
    this->initDescriptors();
}

vk::raii::ShaderModule IQM::GPU::VulkanRuntime::createShaderModule(const uint32_t* spvCode, size_t size) const {
    vk::ShaderModuleCreateInfo shaderModuleCreateInfo{
        .codeSize = size,
        .pCode = spvCode,
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

void IQM::GPU::VulkanRuntime::setImageLayout(const std::shared_ptr<vk::raii::CommandBuffer> &cmd_buf, const vk::raii::Image& image, vk::ImageLayout srcLayout, vk::ImageLayout targetLayout) const {
    vk::AccessFlags sourceAccessMask;
    vk::PipelineStageFlags sourceStage = vk::PipelineStageFlagBits::eTopOfPipe | vk::PipelineStageFlagBits::eTransfer;
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
    return cmd_buf->pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, imageMemoryBarrier);
}

void IQM::GPU::VulkanRuntime::initImages(const std::shared_ptr<vk::raii::CommandBuffer> &cmd_buf, const std::vector<std::shared_ptr<VulkanImage>>& images) {
    vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;

    std::vector<vk::ImageMemoryBarrier> barriers(images.size());

    vk::ImageSubresourceRange imageSubresourceRange(aspectMask, 0, 1, 0, 1);
    for (uint32_t i = 0; i < barriers.size(); i++) {
        barriers[i] = vk::ImageMemoryBarrier{
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = images[i]->image,
            .subresourceRange = imageSubresourceRange
        };
    }
    return cmd_buf->pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,  vk::PipelineStageFlagBits::eTopOfPipe, {}, nullptr, nullptr, barriers);
}

void IQM::GPU::VulkanRuntime::nuke() const {
    auto mask =
        vk::AccessFlagBits::eIndirectCommandRead |
        vk::AccessFlagBits::eIndexRead |
        vk::AccessFlagBits::eVertexAttributeRead |
        vk::AccessFlagBits::eUniformRead |
        vk::AccessFlagBits::eInputAttachmentRead |
        vk::AccessFlagBits::eShaderRead |
        vk::AccessFlagBits::eShaderWrite |
        vk::AccessFlagBits::eColorAttachmentRead |
        vk::AccessFlagBits::eColorAttachmentWrite |
        vk::AccessFlagBits::eDepthStencilAttachmentRead |
        vk::AccessFlagBits::eDepthStencilAttachmentWrite |
        vk::AccessFlagBits::eTransferRead |
        vk::AccessFlagBits::eTransferWrite|
        vk::AccessFlagBits::eHostRead |
        vk::AccessFlagBits::eHostWrite;

    vk::MemoryBarrier barrier {
        .srcAccessMask = mask,
        .dstAccessMask = mask,
    };
    return this->_cmd_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {barrier}, nullptr, nullptr);
}

std::vector<vk::PushConstantRange> IQM::GPU::VulkanRuntime::createPushConstantRange(const unsigned size) {
    return {
        vk::PushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = size,
        }
    };
}

#if PROFILE
void IQM::GPU::VulkanRuntime::createSwapchain(vk::SurfaceKHR surface) {
    uint32_t queues[1] = {this->_queueFamilyIndex};

    auto formats = this->_physicalDevice.getSurfaceFormatsKHR(surface);

    auto cap = this->_physicalDevice.getSurfaceCapabilitiesKHR(surface);

    vk::SwapchainCreateInfoKHR swapchainCreateInfo{
        .surface = surface,
        .minImageCount = cap.minImageCount + 1,
        .imageFormat = formats[0].format,
        .imageExtent = vk::Extent2D{.width = 1280, .height = 720},
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = queues,
        .presentMode = vk::PresentModeKHR::eImmediate,
    };

    this->swapchain = vk::raii::SwapchainKHR{this->_device, swapchainCreateInfo};

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    this->_cmd_buffer->begin(beginInfo);

    vk::AccessFlags sourceAccessMask;
    vk::PipelineStageFlags sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::AccessFlags destinationAccessMask;
    vk::PipelineStageFlags destinationStage = vk::PipelineStageFlagBits::eHost;

    vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;

    vk::ImageSubresourceRange imageSubresourceRange(aspectMask, 0, 1, 0, 1);

    for (const auto image: this->swapchain.getImages()) {
        vk::ImageMemoryBarrier imageMemoryBarrier{
            .srcAccessMask = sourceAccessMask,
            .dstAccessMask = destinationAccessMask,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = imageSubresourceRange
        };
        this->_cmd_buffer->pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, imageMemoryBarrier);
    }

    this->_cmd_buffer->end();

    const std::vector cmdBufs = {
        &**this->_cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{this->_device, vk::FenceCreateInfo{}};

    this->_queue->submit(submitInfo, *fence);
    this->_device.waitIdle();

    this->imageAvailableSemaphore = vk::raii::Semaphore{this->_device, vk::SemaphoreCreateInfo{}};
    this->renderFinishedSemaphore = vk::raii::Semaphore{this->_device, vk::SemaphoreCreateInfo{}};
    this->swapchainFence = vk::raii::Fence{this->_device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}};
}

unsigned IQM::GPU::VulkanRuntime::acquire() {
    auto resWait = this->_device.waitForFences({this->swapchainFence}, true, std::numeric_limits<u_int64_t>::max());
    if (resWait != vk::Result::eSuccess) {
        std::cerr << "Failed to acquire swapchain fence" << std::endl;
    }

    this->_device.resetFences({this->swapchainFence});

    auto[res, index] = this->swapchain.acquireNextImage(std::numeric_limits<u_int64_t>::max(), this->imageAvailableSemaphore, {});
    if (res != vk::Result::eSuccess) {
        std::cerr << "Failed to acquire swapchain image" << std::endl;
    }

    return index;
}

void IQM::GPU::VulkanRuntime::present(unsigned index) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    this->_cmd_buffer->begin(beginInfo);
    this->_cmd_buffer->end();

    const std::vector cmdBufs = {
        &**this->_cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eAllCommands};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*this->imageAvailableSemaphore,
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data(),
    };

    this->_queue->submit(submitInfo, *this->swapchainFence);

    vk::PresentInfoKHR presentInfo{};
    vk::SwapchainKHR swapChains[] = {*this->swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &index;

    auto res = this->_queue->presentKHR(presentInfo);
    if (res != vk::Result::eSuccess) {
        std::cout << "Failed to present" << std::endl;
    }
}
#endif

void IQM::GPU::VulkanRuntime::initQueues() {
    std::optional<vk::raii::PhysicalDevice> physicalDevice;
    // try to access faster dedicated transfer queue
    int computeQueueIndex = -1;
    int transferQueueIndex = -1;

    auto devices = _instance.enumeratePhysicalDevices();
    for (const auto& device : devices) {
        auto properties = device.getProperties();

        physicalDevice = device;

        auto queueFamilyProperties = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilyProperties) {
            if (!(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) && queueFamily.queueFlags & vk::QueueFlagBits::eTransfer && !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
                transferQueueIndex = i;
            }

            if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute && queueFamily.queueFlags & vk::QueueFlagBits::eTransfer && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                computeQueueIndex = i;
            }

            i++;
        }

        this->selectedDevice = std::string(properties.deviceName);
        break;
    }

    this->_physicalDevice = physicalDevice.value();
    this->_queueFamilyIndex = computeQueueIndex;
    this->_transferQueueFamilyIndex = transferQueueIndex;

    float queuePriority = 1.0f;

    std::vector queues = {
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = this->_queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        },
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = this->_transferQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        }
    };

    // no dedicated transfer queue found
    bool dedicatedTransferQueue = true;
    if (this->_transferQueueFamilyIndex == -1) {
        dedicatedTransferQueue = false;
        this->_transferQueueFamilyIndex = this->_queueFamilyIndex;

        queues = {
            vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = this->_queueFamilyIndex,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            },
        };
    }

#ifdef PROFILE
    std::vector deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
#else
    std::vector<char*> deviceExtensions = {};
#endif

    const vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = static_cast<uint32_t>(queues.size()),
        .pQueueCreateInfos = queues.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    this->_device = vk::raii::Device{this->_physicalDevice, deviceCreateInfo};
    this->_queue = std::make_shared<vk::raii::Queue>(this->_device.getQueue(this->_queueFamilyIndex, 0));

    if (dedicatedTransferQueue) {
        this->_transferQueue = std::make_shared<vk::raii::Queue>(this->_device.getQueue(this->_transferQueueFamilyIndex, 0));
    } else {
        this->_transferQueue = this->_queue;
    }

    vk::CommandPoolCreateInfo commandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = static_cast<unsigned>(computeQueueIndex),
    };

    this->_commandPool = std::make_shared<vk::raii::CommandPool>(vk::raii::CommandPool{this->_device, commandPoolCreateInfo});

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
        .commandPool = *this->_commandPool,
        .commandBufferCount = 2,
    };

    auto bufs = vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo};
    this->_cmd_buffer = std::make_shared<vk::raii::CommandBuffer>(std::move(bufs[0]));
    this->_cmd_bufferTransfer = std::make_shared<vk::raii::CommandBuffer>(std::move(bufs[1]));

    if (dedicatedTransferQueue) {
        vk::CommandPoolCreateInfo commandPoolCreateInfoTransfer {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = static_cast<unsigned>(transferQueueIndex),
        };

        this->_commandPoolTransfer = std::make_shared<vk::raii::CommandPool>(vk::raii::CommandPool{this->_device, commandPoolCreateInfoTransfer});

        commandBufferAllocateInfo = {
            .commandPool = *this->_commandPoolTransfer,
            .commandBufferCount = 1,
        };

        this->_cmd_bufferTransfer = std::make_shared<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo}.front()));
    }
}

void IQM::GPU::VulkanRuntime::initDescriptors() {
    this->_descLayoutThreeImage = std::move(this->createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
    }));

    this->_descLayoutTwoImage = std::move(this->createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
    }));

    this->_descLayoutOneImage = std::move(this->createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
    }));

    this->_descLayoutBuffer = std::move(this->createDescLayout({
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    this->_descLayoutImageBuffer = std::move(this->createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    std::vector poolSizes = {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 128},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 32}
    };

    vk::DescriptorPoolCreateInfo dsCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 64,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    this->_descPool = std::move(vk::raii::DescriptorPool{this->_device, dsCreateInfo});
}

std::vector<const char *> IQM::GPU::VulkanRuntime::getLayers() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const auto layer : availableLayers) {
        if (strcmp(layer.layerName,  LAYER_VALIDATION.c_str()) == 0) {
            std::vector layers = {LAYER_VALIDATION.c_str()};
            return layers;
        }
    }

    return {};
}

vk::raii::DescriptorSetLayout IQM::GPU::VulkanRuntime::createDescLayout(const std::vector<std::pair<vk::DescriptorType, uint32_t>> &stub) const {
    auto bindings = std::vector<vk::DescriptorSetLayoutBinding>(stub.size());

    for (unsigned i = 0; i < stub.size(); i++) {
        const auto &[descType, count] = stub[i];
        bindings[i].descriptorCount = count;
        bindings[i].descriptorType = descType;

        // assume only compute stages everywhere
        bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        // recompute indices sequentially
        bindings[i].binding = i;
    }

    auto info = vk::DescriptorSetLayoutCreateInfo {
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return vk::raii::DescriptorSetLayout {this->_device, info};
}

vk::raii::DescriptorSetLayout IQM::GPU::VulkanRuntime::createDescLayout(const std::vector<vk::DescriptorSetLayoutBinding> &bindings) const {
    auto info = vk::DescriptorSetLayoutCreateInfo {
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return vk::raii::DescriptorSetLayout {this->_device, info};
}

std::vector<vk::DescriptorImageInfo> IQM::GPU::VulkanRuntime::createImageInfos(const std::vector<std::shared_ptr<VulkanImage>> &images) {
    std::vector<vk::DescriptorImageInfo> vec(images.size());

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = images[i]->imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
    }

    return vec;
}

void IQM::GPU::VulkanRuntime::waitForFence(const vk::raii::Fence &fence) const {
    auto res = this->_device.waitForFences({fence}, true, std::numeric_limits<uint64_t>::max());
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence");
    }
}

vk::WriteDescriptorSet IQM::GPU::VulkanRuntime::createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorImageInfo> &imgInfos) {
    vk::WriteDescriptorSet writeSet{
        .dstSet = descSet,
        .dstBinding = dstBinding,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<uint32_t>(imgInfos.size()),
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imgInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    return writeSet;
}

vk::WriteDescriptorSet IQM::GPU::VulkanRuntime::createWriteSet(const vk::DescriptorSet &descSet, uint32_t dstBinding, const std::vector<vk::DescriptorBufferInfo> &bufInfos) {
    vk::WriteDescriptorSet writeSet{
        .dstSet = descSet,
        .dstBinding = dstBinding,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<uint32_t>(bufInfos.size()),
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufInfos.data(),
        .pTexelBufferView = nullptr,
    };

    return writeSet;
}


/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vulkan_instance.h"

#include <iostream>

IQM::Profile::VulkanInstance::VulkanInstance(GLFWwindow * window) {
    this->context = vk::raii::Context{};

    vk::ApplicationInfo appInfo{
        .pApplicationName = "Image Quality Metrics",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    auto layers = getLayers();

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

    const vk::InstanceCreateInfo instanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    this->instance = vk::raii::Instance{this->context, instanceCreateInfo};

    this->initQueues();

    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(*this->instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    this->surface = vk::raii::SurfaceKHR{this->instance, vk::SurfaceKHR(surface)};

    this->createSwapchain();
}

void IQM::Profile::VulkanInstance::createSwapchain() {
    uint32_t queues[1] = {this->queueFamilyIndex};

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
    this->cmd_buffer->begin(beginInfo);

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
        this->cmd_buffer->pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, imageMemoryBarrier);
    }

    this->cmd_buffer->end();

    const std::vector cmdBufs = {
        &**this->cmd_buffer
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

unsigned IQM::Profile::VulkanInstance::acquire() {
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

void IQM::Profile::VulkanInstance::present(unsigned index) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    this->cmd_buffer->begin(beginInfo);
    this->cmd_buffer->end();

    const std::vector cmdBufs = {
        &**this->cmd_buffer
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

void IQM::Profile::VulkanInstance::waitForFence(const vk::raii::Fence &fence) const {
    auto res = _device.waitForFences({fence}, true, std::numeric_limits<uint64_t>::max());
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence");
    }
}

void IQM::Profile::VulkanInstance::initQueues() {
    std::optional<vk::raii::PhysicalDevice> physicalDevice;
    // try to access faster dedicated transfer queue
    int computeQueueIndex = -1;
    int transferQueueIndex = -1;

    auto devices = this->instance.enumeratePhysicalDevices();
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
    this->queueFamilyIndex = computeQueueIndex;
    this->transferQueueFamilyIndex = transferQueueIndex;

    float queuePriority = 1.0f;

    std::vector queues = {
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = this->queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        },
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = this->transferQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        }
    };

    // no dedicated transfer queue found
    bool dedicatedTransferQueue = true;
    if (this->transferQueueFamilyIndex == -1) {
        dedicatedTransferQueue = false;
        this->transferQueueFamilyIndex = this->queueFamilyIndex;

        queues = {
            vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = this->queueFamilyIndex,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            },
        };
    }

    std::vector deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    const vk::DeviceCreateInfo deviceCreateInfo{
        .queueCreateInfoCount = static_cast<uint32_t>(queues.size()),
        .pQueueCreateInfos = queues.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    this->_device = vk::raii::Device{this->_physicalDevice, deviceCreateInfo};
    this->_queue = std::make_shared<vk::raii::Queue>(this->_device.getQueue(this->queueFamilyIndex, 0));

    if (dedicatedTransferQueue) {
        this->transferQueue = std::make_shared<vk::raii::Queue>(this->_device.getQueue(this->transferQueueFamilyIndex, 0));
    } else {
        this->transferQueue = this->_queue;
    }

    vk::CommandPoolCreateInfo commandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = static_cast<unsigned>(computeQueueIndex),
    };

    this->commandPool = std::make_shared<vk::raii::CommandPool>(vk::raii::CommandPool{this->_device, commandPoolCreateInfo});

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
        .commandPool = *this->commandPool,
        .commandBufferCount = 2,
    };

    auto bufs = vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo};
    this->cmd_buffer = std::make_shared<vk::raii::CommandBuffer>(std::move(bufs[0]));
    this->cmd_bufferTransfer = std::make_shared<vk::raii::CommandBuffer>(std::move(bufs[1]));

    if (dedicatedTransferQueue) {
        vk::CommandPoolCreateInfo commandPoolCreateInfoTransfer {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = static_cast<unsigned>(transferQueueIndex),
        };

        this->commandPoolTransfer = std::make_shared<vk::raii::CommandPool>(vk::raii::CommandPool{this->_device, commandPoolCreateInfoTransfer});

        commandBufferAllocateInfo = {
            .commandPool = *this->commandPoolTransfer,
            .commandBufferCount = 1,
        };

        this->cmd_bufferTransfer = std::make_shared<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers{this->_device, commandBufferAllocateInfo}.front()));
    }
}

std::vector<const char *> IQM::Profile::VulkanInstance::getLayers() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const auto &layer : availableLayers) {
        if (strcmp(layer.layerName,  LAYER_VALIDATION.c_str()) == 0) {
            std::vector layers = {LAYER_VALIDATION.c_str()};
            return layers;
        }
    }

    return {};
}

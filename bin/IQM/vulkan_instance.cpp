/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "vulkan_instance.h"

IQM::Bin::VulkanInstance::VulkanInstance() {
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
    };

    const vk::InstanceCreateInfo instanceCreateInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    this->instance = vk::raii::Instance{this->context, instanceCreateInfo};

    this->initQueues();
}

void IQM::Bin::VulkanInstance::waitForFence(const vk::raii::Fence &fence) const {
    auto res = _device.waitForFences({fence}, true, std::numeric_limits<uint64_t>::max());
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence");
    }
}

void IQM::Bin::VulkanInstance::initQueues() {
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
    if (this->transferQueueFamilyIndex == static_cast<unsigned>(-1)) {
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

    std::vector<char*> deviceExtensions = {};

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

std::vector<const char *> IQM::Bin::VulkanInstance::getLayers() {
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

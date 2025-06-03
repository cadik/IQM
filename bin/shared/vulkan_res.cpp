/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "vulkan_res.h"

uint32_t IQM::Bin::VulkanResource::findMemoryType(vk::PhysicalDeviceMemoryProperties const &memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask) {
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

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> IQM::Bin::VulkanResource::createBuffer(
    const vk::raii::Device &device,
    const vk::raii::PhysicalDevice &physicalDevice,
    unsigned bufferSize,
    vk::BufferUsageFlags bufferFlags,
    vk::MemoryPropertyFlags memoryFlags) {
    // create now, so it's destroyed before buffer
    vk::raii::DeviceMemory memory{nullptr};

    vk::BufferCreateInfo bufferCreateInfo{
        .size = bufferSize,
        .usage = bufferFlags,
    };

    vk::raii::Buffer buffer{device, bufferCreateInfo};
    auto memReqs = buffer.getMemoryRequirements();
    const auto memType = findMemoryType(
        physicalDevice.getMemoryProperties(),
        memReqs.memoryTypeBits,
        memoryFlags
    );

    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType
    };

    if (memoryFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) {
        allocateSum += memReqs.size;
    }

    memory = vk::raii::DeviceMemory{device, memoryAllocateInfo};

    return std::make_pair(std::move(buffer), std::move(memory));
}

IQM::Bin::VulkanImage IQM::Bin::VulkanResource::createImage(
    const vk::raii::Device &device,
    const vk::raii::PhysicalDevice &physicalDevice,
    const vk::ImageCreateInfo &imageInfo) {
    // create now, so it's destroyed before buffer
    vk::raii::DeviceMemory memory{nullptr};

    vk::raii::Image image{device, imageInfo};
    auto memReqs = image.getMemoryRequirements();
    const auto memType = findMemoryType(
        physicalDevice.getMemoryProperties(),
        memReqs.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = memType
    };

    allocateSum += memReqs.size;

    memory = vk::raii::DeviceMemory{device, memoryAllocateInfo};
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
        .imageView = vk::raii::ImageView{device, imageViewCreateInfo},
        .width = imageInfo.extent.width,
        .height = imageInfo.extent.height,
    };
}

void IQM::Bin::VulkanResource::initImages(const vk::raii::CommandBuffer &cmd_buf, const std::vector<std::shared_ptr<VulkanImage>>& images) {
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
    return cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,  vk::PipelineStageFlagBits::eTopOfPipe, {}, nullptr, nullptr, barriers);
}
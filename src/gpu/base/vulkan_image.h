/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef VULKAN_IMAGE_H
#define VULKAN_IMAGE_H

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace IQM::GPU {

class VulkanImage {
public:
    vk::raii::DeviceMemory memory = VK_NULL_HANDLE;
    vk::raii::Image image = VK_NULL_HANDLE;
    vk::raii::ImageView imageView = VK_NULL_HANDLE;
};

}

#endif //VULKAN_IMAGE_H

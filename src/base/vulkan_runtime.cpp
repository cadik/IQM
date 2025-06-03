/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/base/vulkan_runtime.h>

#include <fstream>
#include <vector>

vk::raii::ShaderModule IQM::GPU::VulkanRuntime::createShaderModule(const vk::raii::Device &device, const std::vector<uint32_t> &spvCode) {
    vk::ShaderModuleCreateInfo shaderModuleCreateInfo{
        .codeSize = spvCode.size() * sizeof(uint32_t),
        .pCode = spvCode.data(),
    };

    vk::raii::ShaderModule module{device, shaderModuleCreateInfo};
    return module;
}

vk::raii::PipelineLayout IQM::GPU::VulkanRuntime::createPipelineLayout(
    const vk::raii::Device &device,
    const std::vector<vk::DescriptorSetLayout> &layouts,
    const std::vector<vk::PushConstantRange> &ranges) {
    vk::PipelineLayoutCreateInfo layoutInfo = {
        .flags = {},
        .setLayoutCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(ranges.size()),
        .pPushConstantRanges = ranges.data(),
    };

    return vk::raii::PipelineLayout{device, layoutInfo};
}

vk::raii::Pipeline IQM::GPU::VulkanRuntime::createComputePipeline(
    const vk::raii::Device &device,
    const vk::raii::ShaderModule &shader,
    const vk::raii::PipelineLayout &layout) {
    vk::ComputePipelineCreateInfo computePipelineCreateInfo{
        .stage = vk::PipelineShaderStageCreateInfo {
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = shader,
            // all shaders will start in main
            .pName = "main",
        },
        .layout = layout
    };

    return std::move(vk::raii::Pipelines{device, nullptr, computePipelineCreateInfo}.front());
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

vk::raii::DescriptorPool IQM::GPU::VulkanRuntime::createDescPool(
    const vk::raii::Device &device,
    uint32_t maxSets,
    std::vector<vk::DescriptorPoolSize> poolSizes)
{
    vk::DescriptorPoolCreateInfo dsCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    return vk::raii::DescriptorPool{device, dsCreateInfo};
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

vk::raii::DescriptorSetLayout IQM::GPU::VulkanRuntime::createDescLayout(const vk::raii::Device &device, const std::vector<std::pair<vk::DescriptorType, uint32_t>> &stub) {
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

    return vk::raii::DescriptorSetLayout {device, info};
}

std::vector<vk::DescriptorImageInfo> IQM::GPU::VulkanRuntime::createImageInfos(const std::vector<const vk::raii::ImageView*> &images) {
    std::vector<vk::DescriptorImageInfo> vec(images.size());

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = *images[i],
            .imageLayout = vk::ImageLayout::eGeneral,
        };
    }

    return vec;
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

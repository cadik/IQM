/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_final_multiply.h"

IQM::GPU::FSIMFinalMultiply::FSIMFinalMultiply(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_final_multiply.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 3,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
    }));

    const std::vector layouts = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);

    // 2x int - index of current execution, buffer size
    const auto sumRanges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));

    this->layout = runtime.createPipelineLayout({this->descSetLayout}, {});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

std::pair<float, float> IQM::GPU::FSIMFinalMultiply::computeMetrics(const VulkanRuntime &runtime, int width, int height) {
    this->prepareImageStorage(runtime, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime.setImageLayout(runtime._cmd_buffer, this->images[0]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->images[1]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->images[2]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    //runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime._device.waitIdle();

    return {0.5, 0.5};
}

void IQM::GPU::FSIMFinalMultiply::prepareImageStorage(const VulkanRuntime &runtime, int width, int height) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D(width, height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    for (unsigned i = 0; i < 3; i++) {
        this->images[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    }

    auto outImageInfos = VulkanRuntime::createImageInfos({this->images[0], this->images[1], this->images[2]});

    const vk::WriteDescriptorSet writeSetOut{
        .dstSet = this->descSet,
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = 3,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = outImageInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetOut,
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

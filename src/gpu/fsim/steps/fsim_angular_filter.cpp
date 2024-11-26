/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_angular_filter.h"

IQM::GPU::FSIMAngularFilter::FSIMAngularFilter(const VulkanRuntime &runtime, const unsigned orientations) : orientations(orientations) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_angular.spv");

    std::vector<vk::DescriptorSetLayout> totalLayouts;
    for (unsigned i = 0; i < orientations; i++) {
        totalLayouts.push_back(*runtime._descLayoutOneImage);
    }

    const std::vector layout = {
        *runtime._descLayoutOneImage,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(totalLayouts.size()),
        .pSetLayouts = totalLayouts.data()
    };

    this->descSets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};

    // 2x int - current orientation, total orientations
    const auto ranges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));

    this->layout = runtime.createPipelineLayout(layout, ranges);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->imageAngularFilters = std::vector<std::shared_ptr<VulkanImage>>(this->orientations);
}

void IQM::GPU::FSIMAngularFilter::constructFilter(const VulkanRuntime &runtime, int width, int height) {
    this->prepareImageStorage(runtime, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);


    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    for (unsigned orientation = 0; orientation < this->orientations; orientation++) {
        runtime.setImageLayout(runtime._cmd_buffer, this->imageAngularFilters[orientation]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSets[orientation]}, {});
        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, orientation);
        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(int), orientations);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    }

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
}

void IQM::GPU::FSIMAngularFilter::prepareImageStorage(const VulkanRuntime &runtime, int width, int height) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D(width, height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    for (unsigned i = 0; i < this->orientations; i++) {
        this->imageAngularFilters[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));

        auto imageInfos = VulkanRuntime::createImageInfos({this->imageAngularFilters[i]});

        const vk::WriteDescriptorSet writeSet{
            .dstSet = this->descSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = imageInfos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        const std::vector writes = {
            writeSet,
        };

        runtime._device.updateDescriptorSets(writes, nullptr);
    }
}

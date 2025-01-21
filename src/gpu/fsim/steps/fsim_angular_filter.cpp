/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_angular_filter.h"

#include <fsim.h>

IQM::GPU::FSIMAngularFilter::FSIMAngularFilter(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_angular.spv");

    std::vector<vk::DescriptorSetLayout> totalLayouts;
    for (unsigned i = 0; i < FSIM_ORIENTATIONS; i++) {
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

    this->imageAngularFilters = std::vector<std::shared_ptr<VulkanImage>>(FSIM_ORIENTATIONS);
}

void IQM::GPU::FSIMAngularFilter::constructFilter(const VulkanRuntime &runtime, int width, int height) {
    this->prepareImageStorage(runtime, width, height);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    for (unsigned orientation = 0; orientation < FSIM_ORIENTATIONS; orientation++) {
        runtime.setImageLayout(runtime._cmd_buffer, this->imageAngularFilters[orientation]->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSets[orientation]}, {});
        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, orientation);
        runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(int), FSIM_ORIENTATIONS);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    }
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

    for (unsigned i = 0; i < FSIM_ORIENTATIONS; i++) {
        this->imageAngularFilters[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));

        auto imageInfos = VulkanRuntime::createImageInfos({this->imageAngularFilters[i]});
        const auto writeSet = VulkanRuntime::createWriteSet(
            this->descSets[i],
            0,
            imageInfos
        );

        const std::vector writes = {
            writeSet,
        };

        runtime._device.updateDescriptorSets(writes, nullptr);
    }
}

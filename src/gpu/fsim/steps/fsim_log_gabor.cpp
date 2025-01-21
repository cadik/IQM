/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_log_gabor.h"

#include <fsim.h>

IQM::GPU::FSIMLogGabor::FSIMLogGabor(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_log_gabor.spv");

    this->descSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
    }));

    const std::vector layout = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layout.size()),
        .pSetLayouts = layout.data()
    };

    this->descSet = std::move(vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo}.front());

    this->layout = runtime.createPipelineLayout(layout, {});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->imageLogGaborFilters = std::vector<std::shared_ptr<VulkanImage>>(FSIM_SCALES);
}

void IQM::GPU::FSIMLogGabor::constructFilter(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height) {
    this->prepareImageStorage(runtime, lowpass, width, height);

    VulkanRuntime::initImages(runtime._cmd_buffer, this->imageLogGaborFilters);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, FSIM_SCALES);
}

void IQM::GPU::FSIMLogGabor::prepareImageStorage(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &lowpass, int width, int height) {
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

    for (unsigned i = 0; i < FSIM_SCALES; i++) {
        this->imageLogGaborFilters[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    }

    auto imageInfosLowpass = VulkanRuntime::createImageInfos({lowpass});
    auto imageInfos = VulkanRuntime::createImageInfos(this->imageLogGaborFilters);

    const auto writeSetLowpass = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfosLowpass
    );

    const auto writeSet = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        imageInfos
    );

    const std::vector writes = {
        writeSetLowpass, writeSet,
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

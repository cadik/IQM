/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_sum_filter_responses.h"

#include <fsim.h>

IQM::GPU::FSIMSumFilterResponses::FSIMSumFilterResponses(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_sum_filter_responses.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageBuffer, 1},
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

    this->layout = runtime.createPipelineLayout(layouts, {});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->filterResponsesInput = std::vector<std::shared_ptr<VulkanImage>>(FSIM_ORIENTATIONS);
    this->filterResponsesRef = std::vector<std::shared_ptr<VulkanImage>>(FSIM_ORIENTATIONS);
}

void IQM::GPU::FSIMSumFilterResponses::computeSums(const VulkanRuntime &runtime, const vk::raii::Buffer &filters, int width, int height) {
    this->prepareImageStorage(runtime, filters, width, height);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    // create only one barrier for all images
    auto images = this->filterResponsesInput;
    images.insert(images.end(),this->filterResponsesRef.begin(),this->filterResponsesRef.end());
    VulkanRuntime::initImages(runtime._cmd_buffer, images);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, FSIM_ORIENTATIONS);
}

void IQM::GPU::FSIMSumFilterResponses::prepareImageStorage(const VulkanRuntime &runtime, const vk::raii::Buffer &filters, int width, int height) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32Sfloat,
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

    for (int i = 0; i < FSIM_ORIENTATIONS; i++) {
        this->filterResponsesInput[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
        this->filterResponsesRef[i] = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    }

    auto imageInfosIn = VulkanRuntime::createImageInfos(this->filterResponsesInput);
    auto imageInfosRef = VulkanRuntime::createImageInfos(this->filterResponsesRef);

    const auto writeSetIn = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfosIn
    );

    const auto writeSetRef = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        imageInfosRef
    );

    const auto bufferInfo = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = filters,
            .offset = 0,
            .range = sizeof(float) * width * height * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    const auto writeSetBuf = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        bufferInfo
    );

    const std::vector writes = {
        writeSetIn, writeSetRef, writeSetBuf
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_phase_congruency.h"

#include <fsim.h>

IQM::GPU::FSIMPhaseCongruency::FSIMPhaseCongruency(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_phase_congruency.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, FSIM_ORIENTATIONS * 2},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS * 2},
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
}

void IQM::GPU::FSIMPhaseCongruency::compute(
    const VulkanRuntime &runtime,
    const vk::raii::Buffer &noiseLevels,
    const std::vector<vk::raii::Buffer> &energyEstimates,
    const std::vector<std::shared_ptr<VulkanImage>> &filterResInput,
    const std::vector<std::shared_ptr<VulkanImage>> &filterResRef,
    int width,
    int height
    ) {
    std::vector<std::shared_ptr<VulkanImage>> filterRes;
    filterRes.insert(filterRes.end(), filterResInput.begin(), filterResInput.end());
    filterRes.insert(filterRes.end(), filterResRef.begin(), filterResRef.end());

    this->prepareImageStorage(runtime, noiseLevels, energyEstimates, filterRes, width, height);

    VulkanRuntime::initImages(runtime._cmd_buffer, {this->pcInput, this->pcRef});

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);
}

void IQM::GPU::FSIMPhaseCongruency::prepareImageStorage(
    const VulkanRuntime &runtime,
    const vk::raii::Buffer &noiseLevels,
    const std::vector<vk::raii::Buffer> &energyEstimates,
    const std::vector<std::shared_ptr<VulkanImage>> &filterRes,
    int width,
    int height
    ) {
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

    this->pcInput = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->pcRef = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));

    auto images = VulkanRuntime::createImageInfos({this->pcInput, this->pcRef});

    const auto writePc = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        images
    );

     auto noiseBuf = std::vector{
         vk::DescriptorBufferInfo {
            .buffer = noiseLevels,
            .offset = 0,
            .range = 2 * FSIM_ORIENTATIONS * sizeof(float),
         }
    };

    const auto writeNoiseLevelsSum = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        noiseBuf
    );

    std::vector<vk::DescriptorBufferInfo> energyBufs(2 * FSIM_ORIENTATIONS);
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        energyBufs[i].buffer = energyEstimates[i];
        energyBufs[i].offset = 0;
        energyBufs[i].range = sizeof(float);
    }

    const auto writeEnergyLevels = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        energyBufs
    );

    auto filterResInfos = VulkanRuntime::createImageInfos(filterRes);

    const auto writeFilterRes = VulkanRuntime::createWriteSet(
        this->descSet,
        3,
        filterResInfos
    );

    const std::vector writes = {
        writePc, writeNoiseLevelsSum, writeEnergyLevels, writeFilterRes
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

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
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = FSIM_ORIENTATIONS * 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = FSIM_ORIENTATIONS * 2,
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

    // 1x int - index
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layout = runtime.createPipelineLayout(layouts, ranges);
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

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    VulkanRuntime::initImages(runtime._cmd_buffer, {this->pcInput, this->pcRef});

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, 0u);
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, 1u);
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime.waitForFence(fence);
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

    const vk::WriteDescriptorSet writePc{
        .dstSet = this->descSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<uint32_t>(images.size()),
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = images.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    vk::DescriptorBufferInfo noiseBuf {
        .buffer = noiseLevels,
        .offset = 0,
        .range = 2 * FSIM_ORIENTATIONS * sizeof(float),
    };

    const vk::WriteDescriptorSet writeNoiseLevelsSum{
        .dstSet = this->descSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &noiseBuf,
        .pTexelBufferView = nullptr,
    };


    std::vector<vk::DescriptorBufferInfo> energyBufs(2 * FSIM_ORIENTATIONS);
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        energyBufs[i].buffer = energyEstimates[i];
        energyBufs[i].offset = 0;
        energyBufs[i].range = sizeof(float);
    }

    const vk::WriteDescriptorSet writeEnergyLevels{
        .dstSet = this->descSet,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<uint32_t>(energyBufs.size()),
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = energyBufs.data(),
        .pTexelBufferView = nullptr,
    };

    auto filterResInfos = VulkanRuntime::createImageInfos(filterRes);

    const vk::WriteDescriptorSet writeFilterRes{
        .dstSet = this->descSet,
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<uint32_t>(filterResInfos.size()),
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = filterResInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writePc, writeNoiseLevelsSum, writeEnergyLevels, writeFilterRes
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_filter_combinations.h"

#include <fsim.h>

IQM::GPU::FSIMFilterCombinations::FSIMFilterCombinations(const VulkanRuntime &runtime) {
    this->multPackKernel = runtime.createShaderModule("../shaders_out/fsim_filter_combinations.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = FSIM_SCALES,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = FSIM_ORIENTATIONS,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = FSIM_SCALES * FSIM_ORIENTATIONS,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
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

    // 1x int - index of current execution
    const auto multPackRanges = VulkanRuntime::createPushConstantRange(1 * sizeof(int));

    this->multPacklayout = runtime.createPipelineLayout({this->descSetLayout}, multPackRanges);
    this->multPackPipeline = runtime.createComputePipeline(this->multPackKernel, this->multPacklayout);

    this->buffers = std::vector<vk::raii::Buffer>();
}

void IQM::GPU::FSIMFilterCombinations::combineFilters(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, int width, int height) {
    this->prepareBufferStorage(runtime, angulars, logGabor, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->multPackPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->multPacklayout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    for (unsigned n = 0; n < FSIM_ORIENTATIONS * FSIM_SCALES; n++) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->multPacklayout, vk::ShaderStageFlagBits::eCompute, 0, n);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    }

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
    runtime._device.waitForFences({fence}, true, std::numeric_limits<uint64_t>::max());
}

void IQM::GPU::FSIMFilterCombinations::prepareBufferStorage(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, int width, int height) {
    uint64_t bufferSize = width * height * sizeof(float) * 2;

    std::vector<vk::DescriptorBufferInfo> bufferInfos;

    for (unsigned n = 0; n < FSIM_ORIENTATIONS * FSIM_SCALES; n++) {
        auto [fftBuf, fftMem] = runtime.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        fftBuf.bindMemory(fftMem, 0);

        bufferInfos.push_back(vk::DescriptorBufferInfo{
            .buffer = fftBuf,
            .offset = 0,
            .range = bufferSize,
        });

        this->buffers.emplace_back(std::move(fftBuf));
        this->memories.emplace_back(std::move(fftMem));
    }

    const vk::WriteDescriptorSet writeSetBuf{
        .dstSet = this->descSet,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS * FSIM_SCALES,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufferInfos.data(),
        .pTexelBufferView = nullptr,
    };

    auto angularInfos = VulkanRuntime::createImageInfos(angulars.imageAngularFilters);
    auto logInfos = VulkanRuntime::createImageInfos(logGabor.imageLogGaborFilters);

    const vk::WriteDescriptorSet writeSetAngular{
        .dstSet = this->descSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = angularInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetLogGabor{
        .dstSet = this->descSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_SCALES,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = logInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetBuf, writeSetAngular, writeSetLogGabor
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

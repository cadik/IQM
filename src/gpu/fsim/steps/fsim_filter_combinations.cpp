/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_filter_combinations.h"

#include <fsim.h>

IQM::GPU::FSIMFilterCombinations::FSIMFilterCombinations(const VulkanRuntime &runtime) {
    this->multPackKernel = runtime.createShaderModule("../shaders_out/fsim_filter_combinations.spv");
    this->sumKernel = runtime.createShaderModule("../shaders_out/fsim_filter_noise.spv");

    //custom layout for this pass
    this->multPackDescSetLayout = std::move(runtime.createDescLayout({
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

    this->sumDescSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = FSIM_SCALES * FSIM_ORIENTATIONS,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }));

    const std::vector layouts = {
        *this->multPackDescSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->multPackDescSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    // 1x int - index of current execution
    const auto multPackRanges = VulkanRuntime::createPushConstantRange(1 * sizeof(int));

    // 2x int - index of current execution, buffer size
    const auto sumRanges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));

    this->multPacklayout = runtime.createPipelineLayout({this->multPackDescSetLayout}, multPackRanges);
    this->multPackPipeline = runtime.createComputePipeline(this->multPackKernel, this->multPacklayout);

    this->sumLayout = runtime.createPipelineLayout({this->sumDescSetLayout}, sumRanges);
    this->sumPipeline = runtime.createComputePipeline(this->sumKernel, this->sumLayout);

    this->buffers = std::vector<vk::raii::Buffer>();
}

void IQM::GPU::FSIMFilterCombinations::combineFilters(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, int width, int height) {
    this->prepareBufferStorage(runtime, angulars, logGabor, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->multPackPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->multPacklayout, 0, {this->multPackDescSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    for (unsigned n = 0; n < FSIM_ORIENTATIONS * FSIM_SCALES; n++) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->multPacklayout, vk::ShaderStageFlagBits::eCompute, 0, n);

        runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
    }

    runtime._cmd_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eDeviceGroup, {}, {}, {});

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint64_t bufferSize = width * height * sizeof(float) * 2;
    for (unsigned n = 0; n < FSIM_ORIENTATIONS; n++) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, n);
        runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, sizeof(unsigned), bufferSize);

        runtime._cmd_buffer->dispatch(1, 1, 1);
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
    runtime.waitForFence(fence);
}

void IQM::GPU::FSIMFilterCombinations::prepareBufferStorage(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, int width, int height) {
    uint64_t noiseLevelsBufferSize = FSIM_ORIENTATIONS * sizeof(float);
    auto [noiseLevelsBuf, noiseLevelsMemory] = runtime.createBuffer(
            noiseLevelsBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    noiseLevelsBuf.bindMemory(noiseLevelsMemory, 0);
    this->noiseLevels = std::move(noiseLevelsBuf);
    this->noiseLevelsMemory = std::move(noiseLevelsMemory);

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
        .dstSet = this->multPackDescSet,
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
        .dstSet = this->multPackDescSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = angularInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetLogGabor{
        .dstSet = this->multPackDescSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_SCALES,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = logInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    vk::DescriptorBufferInfo bufferInfoSum{
        .buffer = this->noiseLevels,
        .offset = 0,
        .range = noiseLevelsBufferSize,
    };

    const vk::WriteDescriptorSet writeSetNoiseDest{
        .dstSet = this->sumDescSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfoSum,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetNoiseSrc{
        .dstSet = this->sumDescSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = FSIM_ORIENTATIONS * FSIM_SCALES,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufferInfos.data(),
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetBuf, writeSetAngular, writeSetLogGabor, writeSetNoiseSrc, writeSetNoiseDest
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

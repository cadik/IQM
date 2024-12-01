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
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
    }));

    this->sumDescSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
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

    // 3x int - buffer size, index of current execution, bool
    const auto sumRanges = VulkanRuntime::createPushConstantRange(3 * sizeof(int));

    this->multPacklayout = runtime.createPipelineLayout({this->multPackDescSetLayout}, multPackRanges);
    this->multPackPipeline = runtime.createComputePipeline(this->multPackKernel, this->multPacklayout);

    this->sumLayout = runtime.createPipelineLayout({this->sumDescSetLayout}, sumRanges);
    this->sumPipeline = runtime.createComputePipeline(this->sumKernel, this->sumLayout);
}

void IQM::GPU::FSIMFilterCombinations::combineFilters(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, const vk::raii::Buffer& fftImages, int width, int height) {
    this->prepareBufferStorage(runtime, angulars, logGabor, fftImages, width, height);

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

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };
    runtime._cmd_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eDeviceGroup, {barrier}, {}, {});

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint64_t bufferSize = width * height * 2;
    runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, bufferSize);

    // parallel sum
    for (unsigned n = 0; n < FSIM_ORIENTATIONS; n++) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, sizeof(unsigned), n);

        vk::BufferCopy region {
            .srcOffset = FSIM_ORIENTATIONS * n * bufferSize * sizeof(float),
            .dstOffset = n * sizeof(float),
            .size = bufferSize * sizeof(float),
        };
        runtime._cmd_buffer->copyBuffer(this->fftBuffer, this->noiseLevels, {region});

        barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        };
        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );
        uint64_t groups = (bufferSize / 128) + 1;
        uint32_t size = bufferSize;
        bool doPower = true;

        for (;;) {
            runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 2 * sizeof(unsigned), doPower);

            runtime._cmd_buffer->dispatch(groups, 1, 1);

            barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
            };
            runtime._cmd_buffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlagBits::eDeviceGroup,
                {barrier},
                {},
                {}
            );
            if (groups == 1) {
                break;
            }
            size = groups;
            groups = (groups / 128) + 1;
            doPower = false;
        }
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

void IQM::GPU::FSIMFilterCombinations::prepareBufferStorage(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, const vk::raii::Buffer& fftImages, int width, int height) {
    uint64_t inFftBufSize = width * height * sizeof(float) * 2 * 2;
    uint64_t outFftBufSize = width * height * sizeof(float) * 2 * FSIM_SCALES * FSIM_ORIENTATIONS * 3;

    // oversize, so parallel sum can be done directly there
    uint64_t noiseLevelsBufferSize = (FSIM_ORIENTATIONS + (width * height * 2 * 2)) * sizeof(float);
    auto [noiseLevelsBuf, noiseLevelsMemory] = runtime.createBuffer(
            noiseLevelsBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    noiseLevelsBuf.bindMemory(noiseLevelsMemory, 0);
    this->noiseLevels = std::move(noiseLevelsBuf);
    this->noiseLevelsMemory = std::move(noiseLevelsMemory);

    auto [fftBuf, fftMem] = runtime.createBuffer(
        outFftBufSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    fftBuf.bindMemory(fftMem, 0);

    this->fftBuffer = std::move(fftBuf);
    this->fftMemory = std::move(fftMem);

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

    vk::DescriptorBufferInfo fftBufInfo{
        .buffer = fftImages,
        .offset = 0,
        .range = inFftBufSize,
    };

    const vk::WriteDescriptorSet writeSetFftIn{
        .dstSet = this->multPackDescSet,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &fftBufInfo,
        .pTexelBufferView = nullptr,
    };

    vk::DescriptorBufferInfo bufferInfo{
        .buffer = this->fftBuffer,
        .offset = 0,
        .range = outFftBufSize,
    };

    const vk::WriteDescriptorSet writeSetBuf{
        .dstSet = this->multPackDescSet,
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfo,
        .pTexelBufferView = nullptr,
    };

    vk::DescriptorBufferInfo bufferInfoSum{
        .buffer = this->noiseLevels,
        .offset = 0,
        .range = noiseLevelsBufferSize,
    };

    const vk::WriteDescriptorSet writeSetNoise{
        .dstSet = this->sumDescSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &bufferInfoSum,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetBuf, writeSetAngular, writeSetLogGabor, writeSetNoise, writeSetFftIn
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

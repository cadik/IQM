/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_estimate_energy.h"

#include <fsim.h>

IQM::GPU::FSIMEstimateEnergy::FSIMEstimateEnergy(const VulkanRuntime &runtime) {
    this->estimateEnergyKernel = runtime.createShaderModule("../shaders_out/fsim_mult_filters.spv");
    this->sumKernel = runtime.createShaderModule("../shaders_out/fsim_noise_energy_sum.spv");

    //custom layout for this pass
    this->estimateEnergyDescSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = FSIM_ORIENTATIONS * 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }));

    this->sumDescSetLayout = std::move(runtime.createDescLayout({
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = FSIM_ORIENTATIONS * 2,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }
    }));

    const std::vector layouts = {
        *this->estimateEnergyDescSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->estimateEnergyDescSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    // 2x int
    const auto estimateEnergyRanges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));

    this->estimateEnergyLayout = runtime.createPipelineLayout({this->estimateEnergyDescSetLayout}, estimateEnergyRanges);
    this->estimateEnergyPipeline = runtime.createComputePipeline(this->estimateEnergyKernel, this->estimateEnergyLayout);

    this->sumLayout = runtime.createPipelineLayout({this->sumDescSetLayout}, estimateEnergyRanges);
    this->sumPipeline = runtime.createComputePipeline(this->sumKernel, this->sumLayout);
}

void IQM::GPU::FSIMEstimateEnergy::estimateEnergy(const VulkanRuntime &runtime, const vk::raii::Buffer &fftBuf, const int width, const int height) {
    this->prepareBufferStorage(runtime, fftBuf, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->estimateEnergyPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->estimateEnergyLayout, 0, {this->estimateEnergyDescSet}, {});
    runtime._cmd_buffer->pushConstants<unsigned>(this->estimateEnergyLayout, vk::ShaderStageFlagBits::eCompute, 0, width * height);

    //shader works in groups of 128 threads
    auto groupsX = ((width * height) / 128) + 1;

    for (int o = 0; o < FSIM_ORIENTATIONS; o++) {
        runtime._cmd_buffer->pushConstants<unsigned>(this->estimateEnergyLayout, vk::ShaderStageFlagBits::eCompute, 1 * sizeof(int), o);

        runtime._cmd_buffer->dispatch(groupsX, 1, 1);
    }

    vk::MemoryBarrier memBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint32_t bufferSize = width * height;
    // now sum
    for (int o = 0; o < FSIM_ORIENTATIONS * 2; o++) {
        uint64_t groups = (bufferSize / 128) + 1;
        uint32_t size = bufferSize;

        for (;;) {
            runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, sizeof(unsigned), o);
            runtime._cmd_buffer->dispatch(groups, 1, 1);

            vk::BufferMemoryBarrier barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .buffer = this->energyBuffers[o],
                .offset = 0,
                .size = bufferSize * sizeof(float),
            };
            runtime._cmd_buffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits::eDeviceGroup,
                {},
                {barrier},
                {}
            );
            if (groups == 1) {
                break;
            }
            size = groups;
            groups = (groups / 128) + 1;
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

void IQM::GPU::FSIMEstimateEnergy::prepareBufferStorage(const VulkanRuntime &runtime, const vk::raii::Buffer &fftBuf, const int width, const int height) {
    uint32_t bufferSize = width * height * sizeof(float);

    this->energyBuffers = std::vector<vk::raii::Buffer>();
    this->energyBuffersMemory = std::vector<vk::raii::DeviceMemory>();
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        auto [buf, mem] = runtime.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        buf.bindMemory(mem, 0);
        this->energyBuffers.emplace_back(std::move(buf));
        this->energyBuffersMemory.emplace_back(std::move(mem));
    }

    const vk::DescriptorBufferInfo fftBufInfo {
        .buffer = fftBuf,
        .offset = 0,
        .range = sizeof(float) * width * height * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
    };

    const vk::WriteDescriptorSet writeSetIn{
        .dstSet = this->estimateEnergyDescSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = &fftBufInfo,
        .pTexelBufferView = nullptr,
    };

    std::vector<vk::DescriptorBufferInfo> outBuffers(2 * FSIM_ORIENTATIONS);
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        outBuffers[i].buffer = this->energyBuffers[i];
        outBuffers[i].offset = 0;
        outBuffers[i].range = bufferSize;
    }

    const vk::WriteDescriptorSet writeSetBuf{
        .dstSet = this->estimateEnergyDescSet,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 2 * FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = outBuffers.data(),
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetSum{
        .dstSet = this->sumDescSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2 * FSIM_ORIENTATIONS,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = outBuffers.data(),
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetIn, writeSetBuf, writeSetSum
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

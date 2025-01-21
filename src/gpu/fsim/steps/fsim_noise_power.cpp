/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "fsim_noise_power.h"

#include <fsim.h>

#include <execution>
#include <algorithm>

static uint32_t src[] =
#include <fsim/fsim_pack_for_median.inc>
;

IQM::GPU::FSIMNoisePower::FSIMNoisePower(const VulkanRuntime &runtime) {
    auto [buf, mem] = runtime.createBuffer(
        2 * FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    buf.bindMemory(mem, 0);
    this->noisePowers = std::move(buf);
    this->noisePowersMemory = std::move(mem);

    this->kernel = runtime.createShaderModule(src, sizeof(src));

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageBuffer, 1},
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

    this->descSet = std::move(vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo}.front());

    // 1x uint - buffer size
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(unsigned));

    this->layout = runtime.createPipelineLayout({this->descSetLayout}, {ranges});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

void IQM::GPU::FSIMNoisePower::copyBackToGpu(const VulkanRuntime &runtime, const vk::raii::Buffer& stgBuf) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = 2 * FSIM_ORIENTATIONS * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(stgBuf, this->noisePowers, {region});

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

void IQM::GPU::FSIMNoisePower::computeNoisePower(const VulkanRuntime &runtime, const vk::raii::Buffer& filterSums, const vk::raii::Buffer& fftBuffer, int width, int height) {
    auto [bufTemp, memTemp] = runtime.createBuffer(
        2 * FSIM_ORIENTATIONS * width * height * sizeof(float),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    bufTemp.bindMemory(memTemp, 0);

    auto bufInfoIn = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = fftBuffer,
            .offset = 0,
            .range = 2 * width * height * sizeof(float) * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    auto bufInfoOut = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = bufTemp,
            .offset = 0,
            .range = width * height * sizeof(float) * FSIM_ORIENTATIONS * 2,
        }
    };

    const auto writeSetIn = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        bufInfoIn
    );

    const auto writeSetOut = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        bufInfoOut
    );

    const std::vector writes = {
        writeSetIn, writeSetOut,
    };

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    runtime._device.updateDescriptorSets(writes, nullptr);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    runtime._cmd_buffer->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, width * height);

    auto groups = (width * height) / 256 + 1;

    runtime._cmd_buffer->dispatch(groups, 1, 1);

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

    auto [stgBuf, stgMem] = runtime.createBuffer(
        2 * FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);
    auto * bufData = static_cast<float*>(stgMem.mapMemory(0,  2 * FSIM_ORIENTATIONS * sizeof(float), {}));

    auto [filterSumsCpuBuf, filterSumsCpuMem] = runtime.createBuffer(
        FSIM_ORIENTATIONS * sizeof(float),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    filterSumsCpuBuf.bindMemory(filterSumsCpuMem, 0);
    this->copyFilterSumsToCpu(runtime, filterSums, filterSumsCpuBuf);

    auto * filterSumsCpu = static_cast<float*>(filterSumsCpuMem.mapMemory(0,  FSIM_ORIENTATIONS * sizeof(float), {}));

    uint64_t largeBufSize = width * height * sizeof(float) * 2 * FSIM_ORIENTATIONS;
    auto [stgBufLarge, stgMemLarge] = runtime.createBuffer(
        largeBufSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBufLarge.bindMemory(stgMemLarge, 0);
    auto * bufDataLarge = static_cast<float*>(stgMemLarge.mapMemory(0, largeBufSize, {}));

    this->copyFilterToCpu(runtime, bufTemp, stgBufLarge, width, height);

    std::vector<float> sortBuf(width * height * 2 * FSIM_ORIENTATIONS);
    memcpy(sortBuf.data(), bufDataLarge, largeBufSize);
    for (int i = 0; i < FSIM_ORIENTATIONS * 2; i++) {
        unsigned int start = i * (width * height);
        unsigned int end = (i + 1) * (width * height);
        std::sort(std::execution::par_unseq, sortBuf.begin() + start, sortBuf.begin() + end);
        float median = sortBuf.at(start + (width * height) / 2);
        float mean = -median / std::log(0.5);

        bufData[i] = mean / filterSumsCpu[i % FSIM_ORIENTATIONS];
    }

    copyBackToGpu(runtime, stgBuf);

    stgMemLarge.unmapMemory();
    stgMem.unmapMemory();
}

void IQM::GPU::FSIMNoisePower::copyFilterToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer& tempBuf, const vk::raii::Buffer& target, int width, int height) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    uint64_t filterSize = width * height * sizeof(float) * 2 * FSIM_ORIENTATIONS;

    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = filterSize,
    };
    runtime._cmd_buffer->copyBuffer(tempBuf, target, {region});

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

void IQM::GPU::FSIMNoisePower::copyFilterSumsToCpu(const VulkanRuntime &runtime, const vk::raii::Buffer &gpuSrc, const vk::raii::Buffer &cpuTarget) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = FSIM_ORIENTATIONS * sizeof(float),
    };
    runtime._cmd_buffer->copyBuffer(gpuSrc, cpuTarget, {region});

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

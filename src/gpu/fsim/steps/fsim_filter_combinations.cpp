/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_filter_combinations.h"

#include <fsim.h>

static uint32_t srcMultPack[] =
#include <fsim/fsim_filter_combinations.inc>
;

static uint32_t srcSum[] =
#include <fsim/fsim_filter_noise.inc>
;

IQM::GPU::FSIMFilterCombinations::FSIMFilterCombinations(const VulkanRuntime &runtime) {
    this->multPackKernel = runtime.createShaderModule(srcMultPack, sizeof(srcMultPack));
    this->sumKernel = runtime.createShaderModule(srcSum, sizeof(srcSum));

    //custom layout for this pass
    this->multPackDescSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, FSIM_SCALES},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    this->sumDescSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageBuffer, 1},
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

    // 3x int - buffer size, index of current execution, bool
    const auto sumRanges = VulkanRuntime::createPushConstantRange(3 * sizeof(int));

    this->multPacklayout = runtime.createPipelineLayout({this->multPackDescSetLayout}, {});
    this->multPackPipeline = runtime.createComputePipeline(this->multPackKernel, this->multPacklayout);

    this->sumLayout = runtime.createPipelineLayout({this->sumDescSetLayout}, sumRanges);
    this->sumPipeline = runtime.createComputePipeline(this->sumKernel, this->sumLayout);
}

void IQM::GPU::FSIMFilterCombinations::combineFilters(const VulkanRuntime &runtime, const FSIMAngularFilter &angulars, const FSIMLogGabor &logGabor, const vk::raii::Buffer& fftImages, int width, int height) {
    this->prepareBufferStorage(runtime, angulars, logGabor, fftImages, width, height);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->multPackPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->multPacklayout, 0, {this->multPackDescSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, FSIM_ORIENTATIONS * FSIM_SCALES);

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

    const auto writeSetAngular = VulkanRuntime::createWriteSet(
        this->multPackDescSet,
        0,
        angularInfos
    );

    const auto writeSetLogGabor = VulkanRuntime::createWriteSet(
        this->multPackDescSet,
        1,
        logInfos
    );

    auto fftBufInfo = std::vector{
        vk::DescriptorBufferInfo{
            .buffer = fftImages,
            .offset = 0,
            .range = inFftBufSize,
        }
    };

    const auto writeSetFftIn = VulkanRuntime::createWriteSet(
        this->multPackDescSet,
        2,
        fftBufInfo
    );

    auto bufferInfo = std::vector{
        vk::DescriptorBufferInfo{
            .buffer = this->fftBuffer,
            .offset = 0,
            .range = outFftBufSize,
        }
    };

    const auto writeSetBuf = VulkanRuntime::createWriteSet(
        this->multPackDescSet,
        3,
        bufferInfo
    );

    auto bufferInfoSum = std::vector{
        vk::DescriptorBufferInfo{
            .buffer = this->noiseLevels,
            .offset = 0,
            .range = noiseLevelsBufferSize,
        }
    };

    const auto writeSetNoise = VulkanRuntime::createWriteSet(
        this->sumDescSet,
        0,
        bufferInfoSum
    );

    const std::vector writes = {
        writeSetBuf, writeSetAngular, writeSetLogGabor, writeSetNoise, writeSetFftIn
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/filter_combinations.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> srcMultPack =
#include <fsim/fsim_filter_combinations.inc>
;

static std::vector<uint32_t> srcSum =
#include <fsim/fsim_filter_noise.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMFilterCombinations::FSIMFilterCombinations(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smMultPack = VulkanRuntime::createShaderModule(device, srcMultPack);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);

    this->multPackDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, FSIM_SCALES},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->sumDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector layouts = {
        *this->multPackDescSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->multPackDescSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    // 3x int - buffer size, index of current execution, bool
    const auto sumRanges = VulkanRuntime::createPushConstantRange(3 * sizeof(int));

    this->multPackLayout = VulkanRuntime::createPipelineLayout(device, {this->multPackDescSetLayout}, {});
    this->multPackPipeline = VulkanRuntime::createComputePipeline(device, smMultPack, this->multPackLayout);

    this->sumLayout = VulkanRuntime::createPipelineLayout(device, {this->sumDescSetLayout}, sumRanges);
    this->sumPipeline = VulkanRuntime::createComputePipeline(device, smSum, this->sumLayout);
}

void IQM::FSIMFilterCombinations::combineFilters(const FSIMInput &input, const unsigned width, const unsigned height, const FftBufferPartitions& partitions) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->multPackPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->multPackLayout, 0, {this->multPackDescSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, FSIM_ORIENTATIONS * FSIM_SCALES);

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };
    input.cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eDeviceGroup, {barrier}, {}, {});

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint64_t bufferSize = width * height * 2;
    input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, bufferSize);

    // parallel sum
    for (unsigned n = 0; n < FSIM_ORIENTATIONS; n++) {
        input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, sizeof(unsigned), n);

        vk::BufferCopy region {
            .srcOffset = FSIM_ORIENTATIONS * n * bufferSize * sizeof(float),
            .dstOffset = n * sizeof(float),
            .size = bufferSize * sizeof(float),
        };
        input.cmdBuf->copyBuffer(*input.bufIfft, **input.bufFft, {region});

        barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        };
        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );
        uint64_t groups = (bufferSize / 1024) + 1;
        uint32_t size = bufferSize;
        bool doPower = true;

        for (;;) {
            input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 2 * sizeof(unsigned), doPower);

            input.cmdBuf->dispatch(groups, 1, 1);

            barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
            };
            input.cmdBuf->pipelineBarrier(
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
            groups = (groups / 1024) + 1;
            doPower = false;
        }
    }

    // copy the summed values into expected position
    vk::BufferCopy region {
        .srcOffset = 0,
        .dstOffset = partitions.noiseLevels,
        .size = FSIM_ORIENTATIONS * sizeof(float),
    };
    input.cmdBuf->copyBuffer(*input.bufFft, **input.bufFft, {region});
    barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );
}

void IQM::FSIMFilterCombinations::setUpDescriptors(const FSIMInput &input, const unsigned width, const unsigned height) {
    uint64_t inFftBufSize = width * height * sizeof(float) * 2 * 2;
    uint64_t outFftBufSize = width * height * sizeof(float) * 2 * FSIM_SCALES * FSIM_ORIENTATIONS * 3;

    // oversize, so parallel sum can be done directly there
    // still smaller than fftBuffer
    uint64_t noiseLevelsBufferSize = (FSIM_ORIENTATIONS + (width * height * 2)) * sizeof(float);

    auto angularInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[4], input.ivFinalSums[0], input.ivFinalSums[1], input.ivFinalSums[2]});
    auto logInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[0], input.ivTempFloat[1], input.ivTempFloat[2], input.ivTempFloat[3]});

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
            .buffer = **input.bufFft,
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
            .buffer = **input.bufIfft,
            .offset = 0,
            .range = outFftBufSize,
        }
    };

    const auto writeSetBuf = VulkanRuntime::createWriteSet(
        this->multPackDescSet,
        3,
        bufferInfo
    );

    // can be reused at this moment
    auto bufferInfoSum = std::vector{
        vk::DescriptorBufferInfo{
            .buffer = **input.bufFft,
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

    input.device->updateDescriptorSets(writes, nullptr);
}

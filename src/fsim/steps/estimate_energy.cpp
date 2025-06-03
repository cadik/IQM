/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/estimate_energy.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> srcMultFilters =
#include <fsim/fsim_mult_filters.inc>
;

static std::vector<uint32_t> srcEnergySum =
#include <fsim/fsim_noise_energy_sum.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMEstimateEnergy::FSIMEstimateEnergy(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smEstimate = VulkanRuntime::createShaderModule(device, srcMultFilters);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcEnergySum);

    this->estimateEnergyDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, FSIM_ORIENTATIONS * 2},
    });

    this->sumDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, FSIM_ORIENTATIONS * 2},
    });

    const std::vector layouts = {
        *this->estimateEnergyDescSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->estimateEnergyDescSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    const auto estimateEnergyRanges = VulkanRuntime::createPushConstantRange(sizeof(int));
    const auto sumRanges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));

    this->estimateEnergyLayout = VulkanRuntime::createPipelineLayout(device, {this->estimateEnergyDescSetLayout}, estimateEnergyRanges);
    this->estimateEnergyPipeline = VulkanRuntime::createComputePipeline(device, smEstimate, this->estimateEnergyLayout);

    this->sumLayout = VulkanRuntime::createPipelineLayout(device, {this->sumDescSetLayout}, sumRanges);
    this->sumPipeline = VulkanRuntime::createComputePipeline(device, smSum, this->sumLayout);
}

void IQM::FSIMEstimateEnergy::estimateEnergy(const FSIMInput& input, const unsigned width, const unsigned height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->estimateEnergyPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->estimateEnergyLayout, 0, {this->estimateEnergyDescSet}, {});
    input.cmdBuf->pushConstants<unsigned>(this->estimateEnergyLayout, vk::ShaderStageFlagBits::eCompute, 0, width * height);

    //shader works in groups of 128 threads
    auto groupsX = ((width * height) / 128) + 1;

    input.cmdBuf->dispatch(groupsX, 1, FSIM_ORIENTATIONS);

    vk::MemoryBarrier memBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint32_t bufferSize = width * height;
    // now sum
    for (int o = 0; o < FSIM_ORIENTATIONS * 2; o++) {
        uint64_t groups = (bufferSize / 1024) + 1;
        uint32_t size = bufferSize;

        for (;;) {
            input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, sizeof(unsigned), o);
            input.cmdBuf->dispatch(groups, 1, 1);

            vk::BufferMemoryBarrier barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .buffer = *input.bufIfft,
                .offset = 0,
                .size = bufferSize * sizeof(float),
            };
            input.cmdBuf->pipelineBarrier(
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
            groups = (groups / 1024) + 1;
        }
    }
}

void IQM::FSIMEstimateEnergy::setUpDescriptors(const FSIMInput& input, const unsigned width, const unsigned height) {
    uint32_t bufferSize = width * height * sizeof(float);

    auto const fftBufInfo = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = **input.bufIfft,
            .offset = 0,
            .range = sizeof(float) * width * height * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    const auto writeSetIn = VulkanRuntime::createWriteSet(
        this->estimateEnergyDescSet,
        0,
        fftBufInfo
    );

    // in this phase of computation, latter 2/3 of iFFT buffer are unused, so reuse them for energy computation
    std::vector<vk::DescriptorBufferInfo> outBuffers(2 * FSIM_ORIENTATIONS);
    uint32_t baseOffset = width * height * sizeof(float) * 2 * FSIM_ORIENTATIONS * FSIM_SCALES;
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        outBuffers[i].buffer = **input.bufIfft;
        outBuffers[i].offset = baseOffset + i * bufferSize;
        outBuffers[i].range = bufferSize;
    }

    const auto writeSetBuf = VulkanRuntime::createWriteSet(
        this->estimateEnergyDescSet,
        1,
        outBuffers
    );

    const auto writeSetSum = VulkanRuntime::createWriteSet(
        this->sumDescSet,
        0,
        outBuffers
    );

    const std::vector writes = {
        writeSetIn, writeSetBuf, writeSetSum
    };

    input.device->updateDescriptorSets(writes, nullptr);
}

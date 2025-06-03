/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/sum_filter_responses.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_sum_filter_responses.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMSumFilterResponses::FSIMSumFilterResponses(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smSum = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector layouts = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);

    this->layout = VulkanRuntime::createPipelineLayout(device, layouts, {});
    this->pipeline = VulkanRuntime::createComputePipeline(device, smSum, this->layout);
}

void IQM::FSIMSumFilterResponses::computeSums(const FSIMInput& input, const unsigned width, const unsigned height) {

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, FSIM_ORIENTATIONS);

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );
}

void IQM::FSIMSumFilterResponses::setUpDescriptors(const FSIMInput& input, const unsigned width, const unsigned height) {
    auto imageInfosIn = VulkanRuntime::createImageInfos(std::vector(std::begin(input.ivFilterResponsesTest), std::end(input.ivFilterResponsesTest)));
    auto imageInfosRef = VulkanRuntime::createImageInfos(std::vector(std::begin(input.ivFilterResponsesRef), std::end(input.ivFilterResponsesRef)));

    const auto writeSetIn = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfosIn
    );

    const auto writeSetRef = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        imageInfosRef
    );

    const auto bufferInfo = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = **input.bufIfft,
            .offset = 0,
            .range = sizeof(float) * width * height * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    const auto writeSetBuf = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        bufferInfo
    );

    const std::vector writes = {
        writeSetIn, writeSetRef, writeSetBuf
    };

    input.device->updateDescriptorSets(writes, nullptr);
}

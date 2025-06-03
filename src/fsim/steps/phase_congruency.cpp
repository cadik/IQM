/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/phase_congruency.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_phase_congruency.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMPhaseCongruency::FSIMPhaseCongruency(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smPc = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, FSIM_ORIENTATIONS * 2},
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS * 2},
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
    this->pipeline = VulkanRuntime::createComputePipeline(device, smPc, this->layout);
}

void IQM::FSIMPhaseCongruency::compute(const FSIMInput &input, const unsigned width, const unsigned height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);
}

void IQM::FSIMPhaseCongruency::setUpDescriptors(const FSIMInput &input, const unsigned width, const unsigned height, const FftBufferPartitions& partitions) const {
    auto images = VulkanRuntime::createImageInfos({input.ivTempFloat[2], input.ivTempFloat[3]});

    const auto writePc = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        images
    );

    auto noiseBuf = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = **input.bufFft,
            .offset = partitions.noisePowers,
            .range = 2 * FSIM_ORIENTATIONS * sizeof(float),
         }
    };

    const auto writeNoiseLevelsSum = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        noiseBuf
    );

    // in this phase of computation, latter 2/3 of iFFT buffer are unused, so reuse them for energy computation
    std::vector<vk::DescriptorBufferInfo> energyBufs(2 * FSIM_ORIENTATIONS);
    uint32_t baseOffset = width * height * sizeof(float) * 2 * FSIM_ORIENTATIONS * FSIM_SCALES;
    uint32_t offset = width * height * sizeof(float);
    for (int i = 0; i < 2 * FSIM_ORIENTATIONS; i++) {
        energyBufs[i].buffer = *input.bufIfft;
        energyBufs[i].offset = baseOffset + i * offset;
        energyBufs[i].range = sizeof(float);
    }

    const auto writeEnergyLevels = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        energyBufs
    );

    std::vector<const vk::raii::ImageView*> filterRes;
    filterRes.insert(filterRes.end(), std::begin(input.ivFilterResponsesTest), std::end(input.ivFilterResponsesTest));
    filterRes.insert(filterRes.end(), std::begin(input.ivFilterResponsesRef), std::end(input.ivFilterResponsesRef));

    auto filterResInfos = VulkanRuntime::createImageInfos(filterRes);

    const auto writeFilterRes = VulkanRuntime::createWriteSet(
        this->descSet,
        3,
        filterResInfos
    );

    const std::vector writes = {
        writePc, writeNoiseLevelsSum, writeEnergyLevels, writeFilterRes
    };

    input.device->updateDescriptorSets(writes, nullptr);
}
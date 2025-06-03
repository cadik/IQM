/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/lpips.h>

static std::vector<uint32_t> srcPreprocess =
#include <lpips/preprocess.inc>
;

static std::vector<uint32_t> srcConv =
#include <lpips/conv.inc>
;

static std::vector<uint32_t> srcConvBig =
#include <lpips/conv_big.inc>
;

static std::vector<uint32_t> srcComapreRelu =
#include <lpips/compare_relu.inc>
;

static std::vector<uint32_t> srcMaxpool =
#include <lpips/maxpool.inc>
;

static std::vector<uint32_t> srcReconstruct =
#include <lpips/reconstruct.inc>
;

static std::vector<uint32_t> srcSum =
#include <lpips/sum.inc>
;

static std::vector<uint32_t> srcPostprocess =
#include <lpips/postprocess.inc>
;

using IQM::GPU::VulkanRuntime;

unsigned dimensionFn(const unsigned size, const unsigned padding, const unsigned kernelSize, const unsigned stride) {
    return (size + 2 * padding - kernelSize) / stride + 1;
}

IQM::LPIPS::LPIPS(const vk::raii::Device &device) {
    const auto smPreprocess = VulkanRuntime::createShaderModule(device, srcPreprocess);
    const auto smConv = VulkanRuntime::createShaderModule(device, srcConv);
    const auto smConvBig = VulkanRuntime::createShaderModule(device, srcConvBig);
    const auto smCompare = VulkanRuntime::createShaderModule(device, srcComapreRelu);
    const auto smMaxpool = VulkanRuntime::createShaderModule(device, srcMaxpool);
    const auto smReconstruct = VulkanRuntime::createShaderModule(device, srcReconstruct);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);
    const auto smPostprocess = VulkanRuntime::createShaderModule(device, srcPostprocess);

    this->descPool = VulkanRuntime::createDescPool(device, 32, {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 80},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 24}
    });

    this->preprocessDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 2},
    });

    this->compareDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->convDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->maxPoolDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->sumDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    std::vector allDescLayouts = {
        *this->preprocessDescSetLayout,
        *this->maxPoolDescSetLayout,
        *this->sumDescSetLayout,
    };

    for (int i = 0; i < 4; i++) {
        allDescLayouts.push_back(*this->maxPoolDescSetLayout);
    }

    for (int i = 0; i < 5; i++) {
        allDescLayouts.push_back(*this->convDescSetLayout);
    }

    for (int i = 0; i < 5; i++) {
        allDescLayouts.push_back(*this->compareDescSetLayout);
    }

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->preprocessDescSet = std::move(sets[0]);
    this->reconstructDescSet = std::move(sets[1]);
    this->sumDescSet = std::move(sets[2]);

    for (int i = 3; i < 7; i++) {
        this->maxPoolDescSets.push_back(std::move(sets[i]));
    }
    for (int i = 7; i < 12; i++) {
        this->convDescSets.push_back(std::move(sets[i]));
    }
    for (int i = 12; i < 17; i++) {
        this->compareDescSets.push_back(std::move(sets[i]));
    }

    this->preprocessLayout = VulkanRuntime::createPipelineLayout(device, {this->preprocessDescSetLayout}, {});
    this->preprocessPipeline = VulkanRuntime::createComputePipeline(device, smPreprocess, this->preprocessLayout);

    const auto convRange = VulkanRuntime::createPushConstantRange(8 * sizeof(uint32_t));
    this->convLayout = VulkanRuntime::createPipelineLayout(device, {this->convDescSetLayout}, {convRange});
    this->createConvPipelines(device, smConv, smConvBig, this->convLayout);

    const auto maxPoolRange = VulkanRuntime::createPushConstantRange(4 * sizeof(uint32_t));
    this->maxPoolLayout = VulkanRuntime::createPipelineLayout(device, {this->maxPoolDescSetLayout}, {maxPoolRange});
    this->maxPoolPipeline = VulkanRuntime::createComputePipeline(device, smMaxpool, this->maxPoolLayout);

    const auto compareRange = VulkanRuntime::createPushConstantRange(3 * sizeof(uint32_t));
    this->compareLayout = VulkanRuntime::createPipelineLayout(device, {this->compareDescSetLayout}, {compareRange});
    this->comparePipeline = VulkanRuntime::createComputePipeline(device, smCompare, this->compareLayout);

    const auto reconstructRange = VulkanRuntime::createPushConstantRange(8 * sizeof(uint32_t));
    this->reconstructLayout = VulkanRuntime::createPipelineLayout(device, {this->maxPoolDescSetLayout}, {reconstructRange});
    this->reconstructPipeline = VulkanRuntime::createComputePipeline(device, smReconstruct, this->reconstructLayout);

    const auto sumRange = VulkanRuntime::createPushConstantRange(1 * sizeof(uint32_t));
    this->sumLayout = VulkanRuntime::createPipelineLayout(device, {this->sumDescSetLayout}, {sumRange});
    this->sumPipeline = VulkanRuntime::createComputePipeline(device, smSum, this->sumLayout);
    this->postprocessPipeline = VulkanRuntime::createComputePipeline(device, smPostprocess, this->sumLayout);
}

void IQM::LPIPS::createConvPipelines(const vk::raii::Device &device, const vk::raii::ShaderModule &sm, const vk::raii::ShaderModule &smBig, const vk::raii::PipelineLayout &layout) {
    unsigned big = this->blocks[0].kernelSize;
    unsigned medium = this->blocks[1].kernelSize;
    unsigned small = this->blocks[2].kernelSize;

    vk::SpecializationMapEntry entry { 0, 0, sizeof(int32_t) };
    vk::SpecializationInfo specInfoBig {
        1,
        &entry,
        sizeof(uint32_t),
        &big,
    };

    vk::SpecializationInfo specInfoMedium {
        1,
        &entry,
        sizeof(uint32_t),
        &medium,
    };

    vk::SpecializationInfo specInfoSmall {
        1,
        &entry,
        sizeof(uint32_t),
        &small,
    };

    std::array createInfos {
        vk::ComputePipelineCreateInfo {
            .stage = vk::PipelineShaderStageCreateInfo {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = smBig,
                // all shaders will start in main
                .pName = "main",
                .pSpecializationInfo = &specInfoBig,
            },
            .layout = layout,
        },
        vk::ComputePipelineCreateInfo {
            .stage = vk::PipelineShaderStageCreateInfo {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = sm,
                // all shaders will start in main
                .pName = "main",
                .pSpecializationInfo = &specInfoMedium,
            },
            .layout = layout,
        },
        vk::ComputePipelineCreateInfo {
            .stage = vk::PipelineShaderStageCreateInfo {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = sm,
                // all shaders will start in main
                .pName = "main",
                .pSpecializationInfo = &specInfoSmall,
            },
            .layout = layout,
        }
    };

    auto pipelines = vk::raii::Pipelines{device, nullptr, createInfos};
    this->convPipelineBig = std::move(pipelines[0]);
    this->convPipelineMedium = std::move(pipelines[1]);
    this->convPipelineSmall = std::move(pipelines[2]);
}


void IQM::LPIPS::computeMetric(const LPIPSInput &input) {
    this->setUpDescriptors(input);

    this->preprocess(input);
    this->conv0(input);
    this->conv1(input);
    this->conv2(input);
    this->conv3(input);
    this->conv4(input);
    this->reconstruct(input);
    this->average(input);
}

void IQM::LPIPS::preprocess(const LPIPSInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->preprocessPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->preprocessLayout, 0, {this->preprocessDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::conv0(const LPIPSInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->convPipelineBig);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->convLayout, 0, {this->convDescSets[0]}, {});
    const auto widthPass = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const std::array pc = {
        input.width,
        input.height,
        widthPass,
        heightPass,
        this->blocks[0].inChannels,
        this->blocks[0].kernelSize,
        this->blocks[0].padding,
        this->blocks[0].stride,
    };
    input.cmdBuf->pushConstants<unsigned>(this->convLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(widthPass, heightPass, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, this->blocks[0].outChannels);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->maxPoolPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->maxPoolLayout, 0, {this->maxPoolDescSets[0]}, {});
    const auto widthMaxPass = dimensionFn(widthPass, 0, 3, 2);
    const auto heightMaxPass = dimensionFn(heightPass, 0, 3, 2);
    const std::array pcMax = {
        widthPass,
        heightPass,
        widthMaxPass,
        heightMaxPass,
    };
    input.cmdBuf->pushConstants<unsigned>(this->maxPoolLayout, vk::ShaderStageFlagBits::eCompute, 0, pcMax);

    auto [groupsMaxX, groupsMaxY] = VulkanRuntime::compute2DGroupCounts(widthMaxPass, heightMaxPass, 16);

    input.cmdBuf->dispatch(groupsMaxX, groupsMaxY, this->blocks[0].outChannels);

    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->maxPoolLayout, 0, {this->maxPoolDescSets[1]}, {});

    input.cmdBuf->dispatch(groupsMaxX, groupsMaxY, this->blocks[0].outChannels);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->comparePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->compareLayout, 0, {this->compareDescSets[0]}, {});
    const std::array pcCompare = {
        widthPass,
        heightPass,
        this->blocks[0].outChannels,
    };
    input.cmdBuf->pushConstants<unsigned>(this->compareLayout, vk::ShaderStageFlagBits::eCompute, 0, pcCompare);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::conv1(const LPIPSInput &input) {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->convPipelineMedium);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->convLayout, 0, {this->convDescSets[1]}, {});

    const std::array pc = {
        widthPass2,
        heightPass2,
        widthPass2,
        heightPass2,
        this->blocks[1].inChannels,
        this->blocks[1].kernelSize,
        this->blocks[1].padding,
        this->blocks[1].stride,
    };
    input.cmdBuf->pushConstants<unsigned>(this->convLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(widthPass2, heightPass2, 4);
    auto [groupsCompareX, groupsCompareY] = VulkanRuntime::compute2DGroupCounts(widthPass2, heightPass2, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, this->blocks[1].outChannels / 64);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->maxPoolPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->maxPoolLayout, 0, {this->maxPoolDescSets[2]}, {});
    const std::array pcMax = {
        widthPass2,
        heightPass2,
        widthPass3,
        heightPass3,
    };
    input.cmdBuf->pushConstants<unsigned>(this->maxPoolLayout, vk::ShaderStageFlagBits::eCompute, 0, pcMax);

    auto [groupsMaxX, groupsMaxY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 16);

    input.cmdBuf->dispatch(groupsMaxX, groupsMaxY, this->blocks[1].outChannels);

    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->maxPoolLayout, 0, {this->maxPoolDescSets[3]}, {});

    input.cmdBuf->dispatch(groupsMaxX, groupsMaxY, this->blocks[1].outChannels);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->comparePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->compareLayout, 0, {this->compareDescSets[1]}, {});
    const std::array pcCompare = {
        widthPass2,
        heightPass2,
        this->blocks[1].outChannels,
    };
    input.cmdBuf->pushConstants<unsigned>(this->compareLayout, vk::ShaderStageFlagBits::eCompute, 0, pcCompare);

    input.cmdBuf->dispatch(groupsCompareX, groupsCompareY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::conv2(const LPIPSInput &input) {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->convPipelineSmall);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->convLayout, 0, {this->convDescSets[2]}, {});

    const std::array pc = {
        widthPass3,
        heightPass3,
        widthPass3,
        heightPass3,
        this->blocks[2].inChannels,
        this->blocks[2].kernelSize,
        this->blocks[2].padding,
        this->blocks[2].stride,
    };
    input.cmdBuf->pushConstants<unsigned>(this->convLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 4);

    input.cmdBuf->dispatch(groupsX, groupsY, this->blocks[2].outChannels / 64);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::conv3(const LPIPSInput &input) {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->convPipelineSmall);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->convLayout, 0, {this->convDescSets[3]}, {});

    const std::array pc = {
        widthPass3,
        heightPass3,
        widthPass3,
        heightPass3,
        this->blocks[3].inChannels,
        this->blocks[3].kernelSize,
        this->blocks[3].padding,
        this->blocks[3].stride,
    };
    input.cmdBuf->pushConstants<unsigned>(this->convLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 4);
    auto [groupsCompareX, groupsCompareY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, this->blocks[3].outChannels / 64);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->comparePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->compareLayout, 0, {this->compareDescSets[2]}, {});
    const std::array pcCompare = {
        widthPass3,
        heightPass3,
        this->blocks[2].outChannels,
    };
    input.cmdBuf->pushConstants<unsigned>(this->compareLayout, vk::ShaderStageFlagBits::eCompute, 0, pcCompare);

    input.cmdBuf->dispatch(groupsCompareX, groupsCompareY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::conv4(const LPIPSInput &input) {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->convPipelineSmall);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->convLayout, 0, {this->convDescSets[4]}, {});

    const std::array pc = {
        widthPass3,
        heightPass3,
        widthPass3,
        heightPass3,
        this->blocks[4].inChannels,
        this->blocks[4].kernelSize,
        this->blocks[4].padding,
        this->blocks[4].stride,
    };
    input.cmdBuf->pushConstants<unsigned>(this->convLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 4);
    auto [groupsCompareX, groupsCompareY] = VulkanRuntime::compute2DGroupCounts(widthPass3, heightPass3, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, this->blocks[4].outChannels / 64);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->comparePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->compareLayout, 0, {this->compareDescSets[3]}, {});
    std::array pcCompare = {
        widthPass3,
        heightPass3,
        this->blocks[3].outChannels,
    };
    input.cmdBuf->pushConstants<unsigned>(this->compareLayout, vk::ShaderStageFlagBits::eCompute, 0, pcCompare);

    input.cmdBuf->dispatch(groupsCompareX, groupsCompareY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->comparePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->compareLayout, 0, {this->compareDescSets[4]}, {});
    pcCompare = {
        widthPass3,
        heightPass3,
        this->blocks[4].outChannels,
    };
    input.cmdBuf->pushConstants<unsigned>(this->maxPoolLayout, vk::ShaderStageFlagBits::eCompute, 0, pcCompare);

    input.cmdBuf->dispatch(groupsCompareX, groupsCompareY, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::reconstruct(const LPIPSInput &input) {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->reconstructPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->reconstructLayout, 0, {this->reconstructDescSet}, {});
    const std::array pc = {
        input.width,
        input.height,
        widthPass1,
        heightPass1,
        widthPass2,
        heightPass2,
        widthPass3,
        heightPass3,
    };
    input.cmdBuf->pushConstants<unsigned>(this->reconstructLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    if (input.imgOut != nullptr) {
        auto region = vk::BufferImageCopy {
            .bufferOffset = 0,
            .bufferRowLength = input.width,
            .bufferImageHeight = input.height,
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{input.width, input.height, 1},
        };

        input.cmdBuf->copyBufferToImage(*input.bufTest, *input.imgOut, vk::ImageLayout::eGeneral, {region});

        memoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        };

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
        );
    }
}

void IQM::LPIPS::average(const LPIPSInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = (input.width) * (input.height);
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;

    for (;;) {
        input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
        input.cmdBuf->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = *input.bufTest,
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
        groups = (groups / sumSize) + 1;
    }

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->postprocessPipeline);
    input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, bufferSize);

    input.cmdBuf->dispatch(1, 1, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::LPIPS::setUpDescriptors(const LPIPSInput &input) const {
    const auto widthPass1 = dimensionFn(input.width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(input.height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    auto writes = std::vector<vk::WriteDescriptorSet>();

    const auto inputImageInfos = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    const auto bufferHalves = this->bufferHalves(input.width, input.height);

    std::vector preBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufTest,
            .offset = 0,
            .range = bufferHalves.input,
        },
        vk::DescriptorBufferInfo{
            .buffer = *input.bufRef,
            .offset = 0,
            .range = bufferHalves.input,
        }
    };

    std::vector convFlipTestBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufTest,
            .offset = 0,
            .range = bufferHalves.input,
        },
    };

    std::vector convFlopTestBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufTest,
            .offset = bufferHalves.input,
            .range = bufferHalves.conv,
        },
    };

    std::vector convFlipRefBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufRef,
            .offset = 0,
            .range = bufferHalves.input,
        },
    };

    std::vector convFlopRefBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufRef,
            .offset = bufferHalves.input,
            .range = bufferHalves.conv,
        },
    };

    std::vector convFlipsBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufTest,
            .offset = 0,
            .range = bufferHalves.input,
        },
        vk::DescriptorBufferInfo{
            .buffer = *input.bufRef,
            .offset = 0,
            .range = bufferHalves.input,
        },
    };

    std::vector convFlopsBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufTest,
            .offset = bufferHalves.input,
            .range = bufferHalves.conv,
        },
        vk::DescriptorBufferInfo{
            .buffer = *input.bufRef,
            .offset = bufferHalves.input,
            .range = bufferHalves.conv,
        },
    };

    unsigned long outAcc = 0;

    std::vector comp0OutBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = 0,
            .range = widthPass1 * heightPass1 * sizeof(float),
        },
    };
    outAcc += widthPass1 * heightPass1 * sizeof(float);
    std::vector comp1OutBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = outAcc,
            .range = widthPass2 * heightPass2 * sizeof(float),
        },
    };
    outAcc += widthPass2 * heightPass2 * sizeof(float);
    std::vector comp2OutBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = outAcc,
            .range = widthPass3 * heightPass3 * sizeof(float),
        },
    };
    outAcc += widthPass3 * heightPass3 * sizeof(float);
    std::vector comp3OutBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = outAcc,
            .range = widthPass3 * heightPass3 * sizeof(float),
        },
    };
    outAcc += widthPass3 * heightPass3 * sizeof(float);
    std::vector comp4OutBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = outAcc,
            .range = widthPass3 * heightPass3 * sizeof(float),
        },
    };
    outAcc += widthPass3 * heightPass3 * sizeof(float);

    std::vector compTotalInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufComp,
            .offset = 0,
            .range = outAcc,
        },
    };

    unsigned long paramsAcc = 0;

    auto conv0weightSize = this->blocks[0].kernelSize * this->blocks[0].kernelSize * this->blocks[0].inChannels * this->blocks[0].outChannels * sizeof(float);
    std::vector conv0WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = 0,
            .range = conv0weightSize,
        },
    };
    paramsAcc += conv0weightSize;

    std::vector conv0BiasBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[0].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[0].outChannels * sizeof(float);

    auto conv1weightSize = this->blocks[1].kernelSize * this->blocks[1].kernelSize * this->blocks[1].inChannels * this->blocks[1].outChannels * sizeof(float);
    std::vector conv1WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = conv1weightSize,
        },
    };
    paramsAcc += conv1weightSize;

    std::vector conv1BiasBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[1].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[1].outChannels * sizeof(float);

    auto conv2weightSize = this->blocks[2].kernelSize * this->blocks[2].kernelSize * this->blocks[2].inChannels * this->blocks[2].outChannels * sizeof(float);
    std::vector conv2WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = conv2weightSize,
        },
    };
    paramsAcc += conv2weightSize;

    std::vector conv2BiasBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[2].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[2].outChannels * sizeof(float);

    auto conv3weightSize = this->blocks[3].kernelSize * this->blocks[3].kernelSize * this->blocks[3].inChannels * this->blocks[3].outChannels * sizeof(float);
    std::vector conv3WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = conv3weightSize,
        },
    };
    paramsAcc += conv3weightSize;

    std::vector conv3BiasBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[3].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[3].outChannels * sizeof(float);

    auto conv4weightSize = this->blocks[4].kernelSize * this->blocks[4].kernelSize * this->blocks[4].inChannels * this->blocks[4].outChannels * sizeof(float);
    std::vector conv4WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = conv4weightSize,
        },
    };
    paramsAcc += conv4weightSize;

    std::vector conv4BiasBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[4].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[4].outChannels * sizeof(float);

    std::vector comp0WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[0].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[0].outChannels * sizeof(float);
    std::vector comp1WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[1].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[1].outChannels * sizeof(float);
    std::vector comp2WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[2].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[2].outChannels * sizeof(float);
    std::vector comp3WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[3].outChannels * sizeof(float),
        },
    };
    paramsAcc += this->blocks[3].outChannels * sizeof(float);
    std::vector comp4WeightBufInfo = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufWeights,
            .offset = paramsAcc,
            .range = this->blocks[4].outChannels * sizeof(float),
        },
    };

    // input
    writes.push_back(VulkanRuntime::createWriteSet(this->preprocessDescSet, 0, inputImageInfos));
    writes.push_back(VulkanRuntime::createWriteSet(this->preprocessDescSet, 1, preBufInfo));

    //conv0
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[0], 0, convFlipsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[0], 1, convFlopsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[0], 2, conv0WeightBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[0], 3, conv0BiasBufInfo));

    //maxpool0
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[0], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[0], 1, convFlipTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[1], 0, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[1], 1, convFlipRefBufInfo));

    //compare0
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[0], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[0], 1, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[0], 2, comp0OutBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[0], 3, comp0WeightBufInfo));

    //conv1
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[1], 0, convFlipsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[1], 1, convFlopsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[1], 2, conv1WeightBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[1], 3, conv1BiasBufInfo));

    //maxpool1
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[2], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[2], 1, convFlipTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[3], 0, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->maxPoolDescSets[3], 1, convFlipRefBufInfo));

    //compare1
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[1], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[1], 1, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[1], 2, comp1OutBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[1], 3, comp1WeightBufInfo));

    //conv2
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[2], 0, convFlipsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[2], 1, convFlopsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[2], 2, conv2WeightBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[2], 3, conv2BiasBufInfo));

    //compare2
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[2], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[2], 1, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[2], 2, comp2OutBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[2], 3, comp2WeightBufInfo));

    //conv3
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[3], 0, convFlopsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[3], 1, convFlipsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[3], 2, conv3WeightBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[3], 3, conv3BiasBufInfo));

    //compare3
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[3], 0, convFlipTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[3], 1, convFlipRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[3], 2, comp3OutBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[3], 3, comp3WeightBufInfo));

    //conv4
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[4], 0, convFlipsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[4], 1, convFlopsBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[4], 2, conv4WeightBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->convDescSets[4], 3, conv4BiasBufInfo));

    //compare4
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[4], 0, convFlopTestBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[4], 1, convFlopRefBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[4], 2, comp4OutBufInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->compareDescSets[4], 3, comp4WeightBufInfo));

    //reconstruct
    writes.push_back(VulkanRuntime::createWriteSet(this->reconstructDescSet, 0, compTotalInfo));
    writes.push_back(VulkanRuntime::createWriteSet(this->reconstructDescSet, 1, convFlipTestBufInfo));

    //sum + postprocess
    writes.push_back(VulkanRuntime::createWriteSet(this->sumDescSet, 0, convFlipTestBufInfo));

    input.device->updateDescriptorSets({writes}, nullptr);
}

IQM::ConvBufferHalves IQM::LPIPS::bufferHalves(unsigned width, unsigned height) const {
    const auto widthPass1 = dimensionFn(width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    const auto inputSize = width * height * 3;
    const auto conv0size = widthPass1 * heightPass1 * this->blocks[0].outChannels;

    return ConvBufferHalves {
        .input = inputSize * sizeof(float),
        .conv = conv0size * sizeof(float),
    };
}

unsigned long IQM::LPIPS::modelSize() const {
    unsigned acc = 0;

    for (const auto & block : this->blocks) {
        // weights, biases, compare
        acc += block.kernelSize * block.kernelSize * block.inChannels * block.outChannels
            + block.outChannels
            + block.outChannels;
    }

    return acc * sizeof(float);
}

IQM::LPIPSBufferSizes IQM::LPIPS::bufferSizes(const unsigned width, const unsigned height) const {
    const auto widthPass1 = dimensionFn(width, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);
    const auto heightPass1 = dimensionFn(height, this->blocks[0].padding, this->blocks[0].kernelSize, this->blocks[0].stride);

    // determined by max pool
    const auto widthPass2 = dimensionFn(widthPass1, 0, 3, 2);
    const auto heightPass2 = dimensionFn(heightPass1, 0, 3, 2);
    const auto widthPass3 = dimensionFn(widthPass2, 0, 3, 2);
    const auto heightPass3 = dimensionFn(heightPass2, 0, 3, 2);

    const auto inputSize = width * height * 3;
    const auto conv0size = widthPass1 * heightPass1 * this->blocks[0].outChannels;

    const auto total = inputSize + conv0size;

    return LPIPSBufferSizes {
        .bufTest = total * sizeof(float),
        .bufRef = total * sizeof(float),
        .bufComp = (widthPass1 * heightPass1 + widthPass2 * heightPass2 + widthPass3 * heightPass3 * 3) * sizeof(float),
        .bufWeights = this->modelSize(),
    };
}
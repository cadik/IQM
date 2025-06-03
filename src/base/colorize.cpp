/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/base/colorize.h>

static std::vector<uint32_t> src =
#include <base/colorize.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::Colorize::Colorize(const vk::raii::Device &device):
descPool(VulkanRuntime::createDescPool(device, 8, {
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 8}
})) {
    const auto sm = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector allDescLayouts = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);

    const auto ranges = VulkanRuntime::createPushConstantRange(2 * sizeof(int));
    this->layout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, ranges);
    this->pipeline = VulkanRuntime::createComputePipeline(device, sm, this->layout);
}

void IQM::Colorize::compute(const ColorizeInput &input) {
    auto imgIn = VulkanRuntime::createImageInfos({input.ivIn});
    auto imgOut = VulkanRuntime::createImageInfos({input.ivOut});
    auto imgColor = VulkanRuntime::createImageInfos({input.ivColormap});

    auto writeSetInput = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imgIn
    );

    auto writeSetOutput = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        imgOut
    );

    auto writeSetColormap = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        imgColor
    );

    input.device->updateDescriptorSets({
        writeSetInput, writeSetOutput, writeSetColormap,
    }, nullptr);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});
    input.cmdBuf->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, input.invert);
    input.cmdBuf->pushConstants<float>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(int), input.scaler);

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
}

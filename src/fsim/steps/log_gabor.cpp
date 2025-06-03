/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/log_gabor.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_log_gabor.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMLogGabor::FSIMLogGabor(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smLogGabor = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
    }));

    const std::vector layout = {
        *this->descSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layout.size()),
        .pSetLayouts = layout.data()
    };

    this->descSet = std::move(vk::raii::DescriptorSets{device, descriptorSetAllocateInfo}.front());

    this->layout = VulkanRuntime::createPipelineLayout(device, layout, {});
    this->pipeline = VulkanRuntime::createComputePipeline(device, smLogGabor, this->layout);
}

void IQM::FSIMLogGabor::setUpDescriptors(const FSIMInput &input) const {
    auto imageInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[0], input.ivTempFloat[1], input.ivTempFloat[2], input.ivTempFloat[3]});

    const std::vector writes = {
        VulkanRuntime::createWriteSet(
            this->descSet,
            0,
            imageInfos
        )
    };

    input.device->updateDescriptorSets(writes, nullptr);
}

void IQM::FSIMLogGabor::constructFilter(const FSIMInput &input, int width, int height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, FSIM_SCALES);
}

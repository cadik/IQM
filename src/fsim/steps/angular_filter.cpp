/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/angular_filter.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_angular.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMAngularFilter::FSIMAngularFilter(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smAngular = VulkanRuntime::createShaderModule(device, src);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, FSIM_ORIENTATIONS},
    });

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
    this->pipeline = VulkanRuntime::createComputePipeline(device, smAngular, this->layout);
}

void IQM::FSIMAngularFilter::constructFilter(const FSIMInput &input, const unsigned width, const unsigned height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, FSIM_ORIENTATIONS);
}

void IQM::FSIMAngularFilter::setUpDescriptors(const FSIMInput &input) const {
    auto imageInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[4], input.ivFinalSums[0], input.ivFinalSums[1], input.ivFinalSums[2]});
    const auto writeSet = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfos
    );

    const std::vector writes = {
        writeSet,
    };

    input.device->updateDescriptorSets(writes, nullptr);
}

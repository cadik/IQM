/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "flip_color_pipeline.h"

IQM::GPU::FLIPColorPipeline::FLIPColorPipeline(const VulkanRuntime &runtime) {
    this->spatialFilterCreateKernel = runtime.createShaderModule("../shaders_out/flip/spatial_filter_create.spv");
    this->spatialFilterNormalizeKernel = runtime.createShaderModule("../shaders_out/flip/spatial_filter_normalize.spv");
    this->csfPrefilterKernel = runtime.createShaderModule("../shaders_out/flip/spatial_prefilter.spv");

    this->spatialFilterCreateDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->csfPrefilterDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageImage, 2},
    });

    const std::vector descLayoutCreate = {
        *this->spatialFilterCreateDescSetLayout,
    };

    const std::vector descLayoutPrefilter = {
        *this->csfPrefilterDescSetLayout,
    };

    const std::vector allDescLayouts = {
        *this->spatialFilterCreateDescSetLayout,
        *this->csfPrefilterDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->spatialFilterCreateDescSet = std::move(sets[0]);
    this->csfPrefilterDescSet = std::move(sets[1]);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float));
    this->spatialFilterCreateLayout = runtime.createPipelineLayout(descLayoutCreate, ranges);
    this->spatialFilterCreatePipeline = runtime.createComputePipeline(this->spatialFilterCreateKernel, this->spatialFilterCreateLayout);
    this->spatialFilterNormalizePipeline = runtime.createComputePipeline(this->spatialFilterNormalizeKernel, this->spatialFilterCreateLayout);

    this->csfPrefilterLayout = runtime.createPipelineLayout(descLayoutPrefilter, {});
    this->csfPrefilterPipeline = runtime.createComputePipeline(this->csfPrefilterKernel, this->csfPrefilterLayout);
}

void IQM::GPU::FLIPColorPipeline::prepareSpatialFilters(const VulkanRuntime &runtime, int kernel_size, float pixels_per_degree) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->spatialFilterCreatePipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->spatialFilterCreateLayout, 0, {this->spatialFilterCreateDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->spatialFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(kernel_size, kernel_size, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->csfFilter->image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, {imageMemoryBarrier}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->spatialFilterNormalizePipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->spatialFilterCreateLayout, 0, {this->spatialFilterCreateDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->spatialFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FLIPColorPipeline::prefilter(const VulkanRuntime &runtime, ImageParameters params) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(params.width, params.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);
}

void IQM::GPU::FLIPColorPipeline::prepareStorage(const VulkanRuntime &runtime, int spatial_kernel_size, ImageParameters params) {
    vk::ImageCreateInfo filterImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = vk::Extent3D(spatial_kernel_size, spatial_kernel_size, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    vk::ImageCreateInfo prefilterImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = vk::Extent3D(params.width, params.height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    this->csfFilter = std::make_shared<VulkanImage>(runtime.createImage(filterImageInfo));
    this->inputPrefilter = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));
    this->refPrefilter = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));

    VulkanRuntime::initImages(runtime._cmd_bufferTransfer, {
        this->csfFilter,
        this->inputPrefilter,
        this->refPrefilter
    });
}

void IQM::GPU::FLIPColorPipeline::setUpDescriptors(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &inputYcc, const std::shared_ptr<VulkanImage> &refYcc) {
    auto imageInfosFilters = VulkanRuntime::createImageInfos({
        this->csfFilter,
    });

    auto imageInfosPrefilterInput = VulkanRuntime::createImageInfos({
        inputYcc,
        refYcc,
    });

    auto imageInfosPrefilterOutput = VulkanRuntime::createImageInfos({
        this->inputPrefilter,
        this->refPrefilter,
    });

    auto writeSetCreate = VulkanRuntime::createWriteSet(
        this->spatialFilterCreateDescSet,
        0,
        imageInfosFilters
    );

    auto writeSetPrefilterInput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        0,
        imageInfosPrefilterInput
    );

    auto writeSetPrefilterFilters = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        1,
        imageInfosFilters
    );

    auto writeSetPrefilterOutput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        2,
        imageInfosPrefilterOutput
    );

    runtime._device.updateDescriptorSets({writeSetCreate, writeSetPrefilterInput, writeSetPrefilterFilters, writeSetPrefilterOutput}, nullptr);
}

/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "flip_color_pipeline.h"

static uint32_t srcHorizontal[] =
#include <flip/spatial_prefilter_horizontal.inc>
;

static uint32_t srcPrefilter[] =
#include <flip/spatial_prefilter.inc>
;

static uint32_t srcDetect[] =
#include <flip/spatial_detection.inc>
;

IQM::GPU::FLIPColorPipeline::FLIPColorPipeline(const VulkanRuntime &runtime) {
    this->csfPrefilterHorizontalKernel = runtime.createShaderModule(srcHorizontal, sizeof(srcHorizontal));
    this->csfPrefilterKernel = runtime.createShaderModule(srcPrefilter, sizeof(srcPrefilter));
    this->spatialDetectKernel = runtime.createShaderModule(srcDetect, sizeof(srcDetect));

    this->csfPrefilterDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
    });

    this->spatialDetectDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector descLayoutPrefilter = {
        *this->csfPrefilterDescSetLayout,
    };

    const std::vector descLayoutDetect = {
        *this->spatialDetectDescSetLayout,
    };

    const std::vector allDescLayouts = {
        *this->csfPrefilterDescSetLayout,
        *this->csfPrefilterDescSetLayout,
        *this->spatialDetectDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->csfPrefilterHorizontalDescSet = std::move(sets[0]);
    this->csfPrefilterDescSet = std::move(sets[1]);
    this->spatialDetectDescSet = std::move(sets[2]);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float));

    this->csfPrefilterLayout = runtime.createPipelineLayout(descLayoutPrefilter, ranges);
    this->csfPrefilterHorizontalPipeline = runtime.createComputePipeline(this->csfPrefilterHorizontalKernel, this->csfPrefilterLayout);
    this->csfPrefilterPipeline = runtime.createComputePipeline(this->csfPrefilterKernel, this->csfPrefilterLayout);

    this->spatialDetectLayout = runtime.createPipelineLayout(descLayoutDetect, {});
    this->spatialDetectPipeline = runtime.createComputePipeline(this->spatialDetectKernel, this->spatialDetectLayout);
}

void IQM::GPU::FLIPColorPipeline::prefilter(const VulkanRuntime &runtime, ImageParameters params, float pixels_per_degree) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterHorizontalPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterHorizontalDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(params.width, params.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);

    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->inputPrefilterTemp->image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vk::ImageMemoryBarrier imageMemoryBarrier_2 = {imageMemoryBarrier};
    imageMemoryBarrier_2.image = this->refPrefilterTemp->image;

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, {imageMemoryBarrier, imageMemoryBarrier_2}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->csfPrefilterPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->csfPrefilterLayout, 0, {this->csfPrefilterDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->csfPrefilterLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);
}

void IQM::GPU::FLIPColorPipeline::computeErrorMap(const VulkanRuntime &runtime, ImageParameters params) {
    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->inputPrefilter->image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    vk::ImageMemoryBarrier imageMemoryBarrier_2 = {imageMemoryBarrier};
    imageMemoryBarrier_2.image = this->refPrefilter->image;

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, {imageMemoryBarrier, imageMemoryBarrier_2}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->spatialDetectPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->spatialDetectLayout, 0, {this->spatialDetectDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(params.width, params.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
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

    vk::ImageCreateInfo colorErrorImageInfo = {prefilterImageInfo};
    colorErrorImageInfo.format = vk::Format::eR32Sfloat;

    this->csfFilter = std::make_shared<VulkanImage>(runtime.createImage(filterImageInfo));
    this->inputPrefilter = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));
    this->refPrefilter = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));
    this->inputPrefilterTemp = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));
    this->refPrefilterTemp = std::make_shared<VulkanImage>(runtime.createImage(prefilterImageInfo));
    this->imageColorError = std::make_shared<VulkanImage>(runtime.createImage(colorErrorImageInfo));

    VulkanRuntime::initImages(runtime._cmd_bufferTransfer, {
        this->csfFilter,
        this->inputPrefilter,
        this->refPrefilter,
        this->imageColorError,
        this->inputPrefilterTemp,
        this->refPrefilterTemp,
    });
}

void IQM::GPU::FLIPColorPipeline::setUpDescriptors(const VulkanRuntime &runtime, const std::shared_ptr<VulkanImage> &inputYcc, const std::shared_ptr<VulkanImage> &refYcc) {
    auto imageInfosPrefilterInput = VulkanRuntime::createImageInfos({
        inputYcc,
        refYcc,
    });

    auto imageInfosPrefilterOutput = VulkanRuntime::createImageInfos({
        this->inputPrefilter,
        this->refPrefilter,
    });

    auto imageInfosPrefilterTemp = VulkanRuntime::createImageInfos({
        this->inputPrefilterTemp,
        this->refPrefilterTemp,
    });

    auto imageInfosOutput = VulkanRuntime::createImageInfos({
        this->imageColorError,
    });

    auto writeSetPrefilterHorInput = VulkanRuntime::createWriteSet(
        this->csfPrefilterHorizontalDescSet,
        0,
        imageInfosPrefilterInput
    );

    auto writeSetPrefilterHorOutput = VulkanRuntime::createWriteSet(
        this->csfPrefilterHorizontalDescSet,
        1,
        imageInfosPrefilterTemp
    );

    auto writeSetPrefilterVertInput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        0,
        imageInfosPrefilterTemp
    );

    auto writeSetPrefilterVertOutput = VulkanRuntime::createWriteSet(
        this->csfPrefilterDescSet,
        1,
        imageInfosPrefilterOutput
    );

    auto writeSetDetectInput = VulkanRuntime::createWriteSet(
        this->spatialDetectDescSet,
        0,
        imageInfosPrefilterOutput
    );

    auto writeSetDetectOutput = VulkanRuntime::createWriteSet(
        this->spatialDetectDescSet,
        1,
        imageInfosOutput
    );

    runtime._device.updateDescriptorSets({writeSetPrefilterHorInput, writeSetPrefilterHorOutput, writeSetPrefilterVertInput, writeSetPrefilterVertOutput, writeSetDetectInput, writeSetDetectOutput}, nullptr);
}

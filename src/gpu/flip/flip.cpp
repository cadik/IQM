/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "flip.h"

IQM::GPU::FLIP::FLIP(const VulkanRuntime &runtime): colorPipeline(runtime) {
    this->inputConvertKernel = runtime.createShaderModule("../shaders_out/flip/srgb_to_ycxcz.spv");
    this->featureFilterCreateKernel = runtime.createShaderModule("../shaders_out/flip/feature_filter.spv");
    this->featureFilterNormalizeKernel = runtime.createShaderModule("../shaders_out/flip/feature_filter_normalize.spv");
    this->featureDetectKernel = runtime.createShaderModule("../shaders_out/flip/feature_detection.spv");

    this->inputConvertDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
    });

    this->featureFilterCreateDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
    });

    this->featureDetectDescSetLayout = runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 1},
    });

    const std::vector descLayouts = {
        *this->inputConvertDescSetLayout,
    };

    const std::vector descLayoutsFeatureFilters = {
        *this->featureFilterCreateDescSetLayout,
    };

    const std::vector descLayoutFilterDetect = {
        *this->featureDetectDescSetLayout,
    };

    const std::vector allDescLayouts = {
        *this->inputConvertDescSetLayout,
        *this->featureFilterCreateDescSetLayout,
        *this->featureDetectDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->inputConvertDescSet = std::move(sets[0]);
    this->featureFilterCreateDescSet = std::move(sets[1]);
    this->featureDetectDescSet = std::move(sets[2]);

    this->inputConvertLayout = runtime.createPipelineLayout(descLayouts, {});
    this->inputConvertPipeline = runtime.createComputePipeline(this->inputConvertKernel, this->inputConvertLayout);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float));
    this->featureFilterCreateLayout = runtime.createPipelineLayout(descLayoutsFeatureFilters, ranges);
    this->featureFilterCreatePipeline = runtime.createComputePipeline(this->featureFilterCreateKernel, this->featureFilterCreateLayout);
    this->featureFilterNormalizePipeline = runtime.createComputePipeline(this->featureFilterNormalizeKernel, this->featureFilterCreateLayout);

    this->featureDetectLayout = runtime.createPipelineLayout(descLayoutFilterDetect, {});
    this->featureDetectPipeline = runtime.createComputePipeline(this->featureDetectKernel, this->featureDetectLayout);

    this->uploadDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->computeDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->transferFence = runtime._device.createFence(vk::FenceCreateInfo{});
}

IQM::GPU::FLIPResult IQM::GPU::FLIP::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref, const FLIPArguments &args) {
    FLIPResult res;

    auto pixels_per_degree = args.monitor_distance * (args.monitor_resolution_x / args.monitor_width) * (std::numbers::pi / 180.0);
    int gaussian_kernel_size = 2 * static_cast<int>(std::ceil(3 * 0.5 * 0.082 * pixels_per_degree)) + 1;
    int spatial_kernel_size = 2 * static_cast<int>(std::ceil(3 * std::sqrt(0.04 / (2.0 * std::pow(std::numbers::pi, 2.0))) * pixels_per_degree)) + 1;

    this->startTransferCommandList(runtime);
    this->prepareImageStorage(runtime, image, ref, gaussian_kernel_size);
    this->colorPipeline.prepareStorage(runtime, spatial_kernel_size, this->imageParameters);
    this->endTransferCommandList(runtime);
    res.timestamps.mark("Image storage prepared");

    this->setUpDescriptors(runtime);
    this->colorPipeline.setUpDescriptors(runtime, this->imageYccInput, this->imageYccRef);
    res.timestamps.mark("Descriptors set up");

    this->convertToYCxCz(runtime);
    this->createFeatureFilters(runtime, pixels_per_degree, gaussian_kernel_size);
    this->colorPipeline.prepareSpatialFilters(runtime, spatial_kernel_size, pixels_per_degree);
    this->computeFeatureErrorMap(runtime);
    this->colorPipeline.prefilter(runtime, this->imageParameters);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*this->uploadDone,
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*this->computeDone
    };

    res.timestamps.mark("GPU work prepared");

    runtime._queue->submit(submitInfo, {});
    runtime._device.waitIdle();
    //runtime.waitForFence(this->transferFence);

    res.timestamps.mark("GPU work done");

    return res;
}

void IQM::GPU::FLIP::prepareImageStorage(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref, int kernel_size) {
    // always 4 channels on input, with 1B per channel
    const auto size = image.width * image.height * 4;
    auto [stgBuf, stgMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [stgRefBuf, stgRefMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    this->imageParameters.height = image.height;
    this->imageParameters.width = image.width;

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, image.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    this->stgInput = std::move(stgBuf);
    this->stgInputMemory = std::move(stgMem);
    this->stgRef = std::move(stgRefBuf);
    this->stgRefMemory = std::move(stgRefMem);

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(image.width, image.height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    vk::ImageCreateInfo yccImageInfo {srcImageInfo};
    yccImageInfo.format = vk::Format::eR32G32B32A32Sfloat;
    yccImageInfo.usage = vk::ImageUsageFlagBits::eStorage;

    vk::ImageCreateInfo errorImageInfo {srcImageInfo};
    errorImageInfo.format = vk::Format::eR32Sfloat;
    errorImageInfo.usage = vk::ImageUsageFlagBits::eStorage;

    vk::ImageCreateInfo featureFilterImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D(kernel_size, kernel_size, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    this->imageInput = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageRef = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageYccInput = std::make_shared<VulkanImage>(runtime.createImage(yccImageInfo));
    this->imageYccRef = std::make_shared<VulkanImage>(runtime.createImage(yccImageInfo));
    this->imageFeaturePointFilter = std::make_shared<VulkanImage>(runtime.createImage(featureFilterImageInfo));
    this->imageFeatureEdgeFilter = std::make_shared<VulkanImage>(runtime.createImage(featureFilterImageInfo));
    this->imageFeatureError = std::make_shared<VulkanImage>(runtime.createImage(errorImageInfo));

    VulkanRuntime::initImages(runtime._cmd_bufferTransfer, {
        this->imageInput,
        this->imageRef,
        this->imageYccInput,
        this->imageYccRef,
        this->imageFeaturePointFilter,
        this->imageFeatureEdgeFilter,
        this->imageFeatureError,
    });

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyBufferToImage(this->stgInput, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_bufferTransfer->copyBufferToImage(this->stgRef, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);
}

void IQM::GPU::FLIP::convertToYCxCz(const VulkanRuntime &runtime) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->inputConvertPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->inputConvertLayout, 0, {this->inputConvertDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(this->imageParameters.width, this->imageParameters.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);
}

void IQM::GPU::FLIP::createFeatureFilters(const VulkanRuntime &runtime, float pixels_per_degree, int kernel_size) {
    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterCreatePipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(kernel_size, kernel_size, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);

    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->imageFeatureEdgeFilter->image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    vk::ImageMemoryBarrier imageMemoryBarrier_2 = {imageMemoryBarrier};
    imageMemoryBarrier_2.image = this->imageFeaturePointFilter->image;

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, {imageMemoryBarrier, imageMemoryBarrier_2}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterNormalizePipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    runtime._cmd_buffer->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixels_per_degree);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 2);
}

void IQM::GPU::FLIP::computeFeatureErrorMap(const VulkanRuntime &runtime) {
    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->imageFeatureEdgeFilter->image,
        .subresourceRange = vk::ImageSubresourceRange {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    vk::ImageMemoryBarrier imageMemoryBarrier_2 = {imageMemoryBarrier};
    imageMemoryBarrier_2.image = this->imageFeaturePointFilter->image;
    vk::ImageMemoryBarrier imageMemoryBarrierInput = {imageMemoryBarrier};
    imageMemoryBarrierInput.image = this->imageYccInput->image;
    vk::ImageMemoryBarrier imageMemoryBarrierRef = {imageMemoryBarrier};
    imageMemoryBarrierRef.image = this->imageYccRef->image;

    // wait here, so previous work can be run in parallel
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, {imageMemoryBarrier, imageMemoryBarrier_2, imageMemoryBarrierInput, imageMemoryBarrierRef}
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureDetectPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureDetectLayout, 0, {this->featureDetectDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(this->imageParameters.width, this->imageParameters.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FLIP::startTransferCommandList(const VulkanRuntime &runtime) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfo);
}

void IQM::GPU::FLIP::endTransferCommandList(const VulkanRuntime &runtime) {
    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    const vk::SubmitInfo submitInfoCopy{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*this->uploadDone
    };

    runtime._transferQueue->submit(submitInfoCopy, this->transferFence);
}

void IQM::GPU::FLIP::setUpDescriptors(const VulkanRuntime &runtime) {
    auto imageInfos = VulkanRuntime::createImageInfos({
        this->imageInput,
        this->imageRef,
    });

    auto writeSetConvertInput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        0,
        imageInfos
    );

    auto outImageInfos = VulkanRuntime::createImageInfos({
        this->imageYccInput,
        this->imageYccRef,
    });

    auto writeSetConvertOutput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        1,
        outImageInfos
    );

    auto featureFilterImageInfos = VulkanRuntime::createImageInfos({
        this->imageFeatureEdgeFilter,
        this->imageFeaturePointFilter,
    });

    auto writeSetFeatureFilter = VulkanRuntime::createWriteSet(
        this->featureFilterCreateDescSet,
        0,
        featureFilterImageInfos
    );

    auto outFeatureImageInfos = VulkanRuntime::createImageInfos({
        this->imageFeatureError,
    });

    auto writeSetDetectInput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        0,
        outImageInfos
    );

    auto writeSetDetectFilters = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        1,
        featureFilterImageInfos
    );

    auto writeSetDetectOutput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        2,
        outFeatureImageInfos
    );

    runtime._device.updateDescriptorSets({writeSetConvertInput, writeSetConvertOutput, writeSetFeatureFilter, writeSetDetectInput, writeSetDetectFilters, writeSetDetectOutput}, nullptr);
}

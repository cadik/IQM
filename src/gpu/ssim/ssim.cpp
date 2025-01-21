/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "ssim.h"

static uint32_t src[] =
#include <ssim/ssim.inc>
;

static uint32_t srcLumapack[] =
#include <ssim/ssim_lumapack.inc>
;

static uint32_t srcGaussInput[] =
#include <ssim/ssim_gaussinput.inc>
;

IQM::GPU::SSIM::SSIM(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule(src, sizeof(src));
    this->kernelLumapack = runtime.createShaderModule(srcLumapack, sizeof(srcLumapack));
    this->kernelGaussInput = runtime.createShaderModule(srcGaussInput, sizeof(srcGaussInput));

    const std::vector layouts_3 = {
        *runtime._descLayoutThreeImage
    };

    const std::vector layouts_2 = {
        *runtime._descLayoutTwoImage
    };

    const std::vector allocateLayouts = {
        *runtime._descLayoutThreeImage,
        *runtime._descLayoutThreeImage,
        *runtime._descLayoutTwoImage
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allocateLayouts.size()),
        .pSetLayouts = allocateLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSetLumapack = std::move(sets[0]);
    this->descSet = std::move(sets[1]);
    this->descSetGaussInput = std::move(sets[2]);

    // 1x int - kernel size
    // 3x float - K_1, K_2, sigma
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1 + sizeof(float) * 3);

    // 1x int - kernel size
    // 1x float - sigma
    const auto rangesGauss = VulkanRuntime::createPushConstantRange(sizeof(int) + sizeof(float));

    this->layout = runtime.createPipelineLayout(layouts_3, ranges);
    this->layoutLumapack = runtime.createPipelineLayout(layouts_3, {});
    this->layoutGaussInput = runtime.createPipelineLayout(layouts_2, rangesGauss);

    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
    this->pipelineLumapack = runtime.createComputePipeline(this->kernelLumapack, this->layoutLumapack);
    this->pipelineGaussInput = runtime.createComputePipeline(this->kernelGaussInput, this->layoutGaussInput);

    this->uploadDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->computeDone = runtime._device.createSemaphore(vk::SemaphoreCreateInfo{});
    this->transferFence = runtime._device.createFence(vk::FenceCreateInfo{});
}

IQM::GPU::SSIMResult IQM::GPU::SSIM::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    runtime._device.resetFences({this->transferFence});

    this->prepareImages(runtime, image, ref);

    SSIMResult res;

    res.timestamps.mark("start GPU pipeline");

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineLumapack);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutLumapack, 0, {this->descSetLumapack}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(this->imageParameters.width, this->imageParameters.height, 16);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::ImageMemoryBarrier imageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->imageLuma->image,
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
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, imageMemoryBarrier
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGaussInput);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGaussInput, 0, {this->descSetGaussInput}, {});

    std::array valuesGauss = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->sigma)
    };
    runtime._cmd_buffer->pushConstants<int>(this->layoutGaussInput, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::ImageMemoryBarrier gaussImageMemoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = this->imageLumaBlurred->image,
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
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, gaussImageMemoryBarrier
    );

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    std::array values = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->k_1),
        *reinterpret_cast<int *>(&this->k_2),
        *reinterpret_cast<int *>(&this->sigma)
    };
    runtime._cmd_buffer->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, values);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

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

    runtime._queue->submit(submitInfo, {});
    runtime.waitForFence(this->transferFence);

    const auto size = this->imageParameters.height * this->imageParameters.width * sizeof(float);
    auto [stgBuf, stgMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
    );
    stgBuf.bindMemory(stgMem, 0);

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyImageToBuffer(this->imageOut->image,  vk::ImageLayout::eGeneral, stgBuf, copyRegion);

    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*this->computeDone,
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._transferQueue->submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();

    res.timestamps.mark("end GPU pipeline");

    std::vector<float> outputData(this->imageParameters.height * this->imageParameters.width);
    void * outBufData = stgMem.mapMemory(0, this->imageParameters.height * this->imageParameters.width * sizeof(float), {});
    memcpy(outputData.data(), outBufData, this->imageParameters.height * this->imageParameters.width * sizeof(float));
    res.timestamps.mark("end copy from GPU");

    res.mssim = this->computeMSSIM( static_cast<float*>(outBufData), this->imageParameters.width, this->imageParameters.height);
    res.timestamps.mark("end MSSIM compute");
    stgMem.unmapMemory();

    res.imageData = std::move(outputData);
    res.height = this->imageParameters.height;
    res.width = this->imageParameters.width;

    return res;
}

void IQM::GPU::SSIM::prepareImages(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
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

    vk::ImageCreateInfo lumaImageInfo {srcImageInfo};
    lumaImageInfo.format = vk::Format::eR32G32Sfloat;
    lumaImageInfo.usage = vk::ImageUsageFlagBits::eStorage;

    vk::ImageCreateInfo dstImageInfo = {srcImageInfo};
    dstImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    dstImageInfo.format = vk::Format::eR32Sfloat;

    this->imageInput = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageRef = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageOut = std::make_shared<VulkanImage>(runtime.createImage(dstImageInfo));
    this->imageLuma = std::make_shared<VulkanImage>(runtime.createImage(lumaImageInfo));
    this->imageLumaBlurred = std::make_shared<VulkanImage>(runtime.createImage(lumaImageInfo));

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfo);

    VulkanRuntime::initImages(runtime._cmd_bufferTransfer, {
        this->imageInput,
        this->imageRef,
        this->imageOut,
        this->imageLuma,
        this->imageLumaBlurred,
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

    auto imageInfos = VulkanRuntime::createImageInfos({
        this->imageLuma,
        this->imageLumaBlurred,
        this->imageOut,
    });

    auto writeSet = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        imageInfos
    );

    auto lumapackImageInfos = VulkanRuntime::createImageInfos({
        this->imageInput,
        this->imageRef,
        this->imageLuma,
    });

    auto writeSetLumapack = VulkanRuntime::createWriteSet(
        this->descSetLumapack,
        0,
        lumapackImageInfos
    );

    auto gaussInputImageInfos = VulkanRuntime::createImageInfos({
        this->imageLuma,
        this->imageLumaBlurred,
    });

    auto writeSetGauss = VulkanRuntime::createWriteSet(
        this->descSetGaussInput,
        0,
        gaussInputImageInfos
    );

    runtime._device.updateDescriptorSets({writeSet, writeSetLumapack, writeSetGauss}, nullptr);
}

double IQM::GPU::SSIM::computeMSSIM(const float* buffer, unsigned width, unsigned height) const {
    // there are two passes of gaussian blur, original MATLAB code trims the boundary of images
    // so that zero padded edges are not included in the final computation
    double sum = 0;
    const unsigned start = (this->kernelSize - 1) / 2;
    const unsigned widthEnd = width - (this->kernelSize - 1) / 2;
    const unsigned heightEnd = height - (this->kernelSize - 1) / 2;

    for (unsigned y = start; y <= heightEnd; y++) {
        for (unsigned x = start; x <= widthEnd; x++) {
            sum += buffer[y * width + x];
        }
    }

    return sum / static_cast<double>((widthEnd - start + 1) * (heightEnd - start + 1));
}

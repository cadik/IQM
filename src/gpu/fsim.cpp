#include "fsim.h"

#include "img_params.h"

IQM::GPU::FSIM::FSIM(const VulkanRuntime &runtime) {
    this->downscaleKernel = runtime.createShaderModule("../shaders_out/fsim_downsample.spv");

    const std::vector layouts = {
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSetDownscaleIn = std::move(sets[0]);
    this->descSetDownscaleRef = std::move(sets[1]);

    // 2x int - kernel size, retyped bool
    std::vector ranges {
        vk::PushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(int) * 2,
        }
    };

    this->layoutDownscale = runtime.createPipelineLayout(layouts, ranges);
    this->pipelineDownscale = runtime.createComputePipeline(this->downscaleKernel, this->layoutDownscale);
}

cv::Mat IQM::GPU::FSIM::computeMetric(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref) {
    assert(image.rows == ref.rows);
    assert(image.cols == ref.cols);

    const int F = computeDownscaleFactor(image.cols, image.rows);

    this->sendImagesToGpu(runtime, image, ref);

    const auto widthDownscale = static_cast<int>(std::round(static_cast<float>(image.cols) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<int>(std::round(static_cast<float>(image.rows) / static_cast<float>(F)));

    this->createDownscaledImages(runtime, widthDownscale, heightDownscale);
    this->computeDownscaledImages(runtime, F, widthDownscale, heightDownscale);

    return image;
}

int IQM::GPU::FSIM::computeDownscaleFactor(const int cols, const int rows) {
    auto smallerDim = std::min(cols, rows);
    return std::max(1, static_cast<int>(std::round(smallerDim / 256.0)));
}

void IQM::GPU::FSIM::sendImagesToGpu(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref) {
    const auto size = image.rows * image.cols * image.channels();
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

    auto imageParameters = ImageParameters(image.cols, image.rows);

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, image.data, imageParameters.height * imageParameters.width * 4);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, ref.data, imageParameters.height * imageParameters.width * 4);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(image.cols, image.rows, 1),
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

    this->imageInput = runtime.createImage(srcImageInfo);
    this->imageRef = runtime.createImage(srcImageInfo);

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInput.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRef.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = imageParameters.width,
        .bufferImageHeight = imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{imageParameters.width, imageParameters.height, 1}
    };
    runtime._cmd_buffer.copyBufferToImage(stgBuf, this->imageInput.image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_buffer.copyBufferToImage(stgRefBuf, this->imageRef.image,  vk::ImageLayout::eGeneral, copyRegion);

    runtime._cmd_buffer.end();

    const std::vector cmdBufsCopy = {
        &*runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eAllCommands};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue.submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();
}

void IQM::GPU::FSIM::createDownscaledImages(const VulkanRuntime &runtime, int width_downscale, int height_downscale) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = vk::Extent3D(width_downscale, height_downscale, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    this->imageInputDownscaled = runtime.createImage(imageInfo);
    this->imageRefDownscaled = runtime.createImage(imageInfo);

    std::vector imageInfosInput = {
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageInput.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageInputDownscaled.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
    };

    std::vector imageInfosRef = {
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageRef.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageRefDownscaled.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
    };

    const vk::WriteDescriptorSet writeSetInput{
        .dstSet = this->descSetDownscaleIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosInput.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetRef{
        .dstSet = this->descSetDownscaleRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetRef, writeSetInput
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

void IQM::GPU::FSIM::computeDownscaledImages(const VulkanRuntime &runtime, const int F, const int width, const int height) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInputDownscaled.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRefDownscaled.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineDownscale);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleIn}, {});

    int useColor = this->doColorComparison;

    std::array values = {
        F,
        useColor,
    };
    runtime._cmd_buffer.pushConstants<int>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, values);

    //shader works in 16x16 tiles
    constexpr unsigned tileSize = 8;

    auto groupsX = width / tileSize;
    if (width % tileSize != 0) {
        groupsX++;
    }
    auto groupsY = height / tileSize;
    if (height % tileSize != 0) {
        groupsY++;
    }

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleRef}, {});
    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer.end();

    const std::vector cmdBufs = {
        &*runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue.submit(submitInfo, *fence);
    runtime._device.waitIdle();
}

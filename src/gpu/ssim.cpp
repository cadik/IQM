#include "ssim.h"

IQM::GPU::SSIM::SSIM(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/ssim.spv");

    const std::vector layouts = {
        *runtime._descLayout
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts.data()
    };

    this->descSet = std::move(vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo}.front());

    vk::PipelineLayoutCreateInfo layoutInfo = {
        .flags = {},
        .setLayoutCount = 1,
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = {},
    };

    this->layout = runtime.createPipelineLayout(layoutInfo);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

cv::Mat IQM::GPU::SSIM::computeMetric(const VulkanRuntime &runtime) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    //shader works in 16x16 tiles
    constexpr unsigned tileSize = 16;

    auto groupsX = this->imageParameters.width / tileSize;
    if (this->imageParameters.width % tileSize != 0) {
        groupsX++;
    }
    auto groupsY = this->imageParameters.height / tileSize;
    if (this->imageParameters.height % tileSize != 0) {
        groupsY++;
    }

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

    const auto size = this->imageParameters.height * this->imageParameters.width * 4;
    auto [stgBuf, stgMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);

    // copy out
    runtime._cmd_buffer.reset();
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_buffer.copyImageToBuffer(this->imageOutput,  vk::ImageLayout::eGeneral, stgBuf, copyRegion);

    runtime._cmd_buffer.end();

    const std::vector cmdBufsCopy = {
        &*runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue.submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();

    cv::Mat dummy;
    dummy.create(static_cast<int>(this->imageParameters.height), static_cast<int>(this->imageParameters.width), CV_8UC4);
    void * outBufData = stgMem.mapMemory(0, this->imageParameters.height * this->imageParameters.width * 4, {});
    memcpy(dummy.data, outBufData, this->imageParameters.height * this->imageParameters.width * 4);
    stgMem.unmapMemory();

    return dummy;
}

void IQM::GPU::SSIM::prepareImages(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref) {
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

    assert(image.rows == ref.rows);
    assert(image.cols == ref.cols);
    this->imageParameters.height = image.rows;
    this->imageParameters.width = image.cols;

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

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

    vk::ImageCreateInfo dstImageInfo = {srcImageInfo};
    dstImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;

    auto [imgInput, imgInputMem] = runtime.createImage(srcImageInfo);
    this->imageInput = std::move(imgInput);
    this->imageInputMemory = std::move(imgInputMem);
    this->imageInputView = std::move(runtime.createImageView(this->imageInput));

    auto [imgRef, imgRefMem] = runtime.createImage(srcImageInfo);
    this->imageRef = std::move(imgRef);
    this->imageRefMemory = std::move(imgRefMem);
    this->imageRefView = std::move(runtime.createImageView(this->imageRef));

    auto [imgOut, imgOutMem] = runtime.createImage(dstImageInfo);
    this->imageOutput = std::move(imgOut);
    this->imageOutputMemory = std::move(imgOutMem);
    this->imageOutputView = std::move(runtime.createImageView(this->imageOutput));

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInput, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRef, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageOutput, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

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

    std::vector imageInfos = {
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageInputView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageRefView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
        vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = this->imageOutputView,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
    };
    
    vk::WriteDescriptorSet writeSet{
        .dstSet = this->descSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 3,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    runtime._device.updateDescriptorSets(writeSet, nullptr);
}

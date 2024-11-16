#include "ssim.h"

IQM::GPU::SSIM::SSIM(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/ssim.spv");
    this->kernelLumapack = runtime.createShaderModule("../shaders_out/ssim_lumapack.spv");
    this->kernelGaussInput = runtime.createShaderModule("../shaders_out/ssim_gaussinput.spv");

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
    std::vector ranges {
        vk::PushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(int) * 1 + sizeof(float) * 3,
        }
    };

    // 1x int - kernel size
    // 1x float - sigma
    std::vector rangesGauss {
        vk::PushConstantRange {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(int) * 1 + sizeof(float) * 1,
        }
    };

    this->layout = runtime.createPipelineLayout(layouts_3, ranges);
    this->layoutLumapack = runtime.createPipelineLayout(layouts_3, {});
    this->layoutGaussInput = runtime.createPipelineLayout(layouts_2, rangesGauss);

    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
    this->pipelineLumapack = runtime.createComputePipeline(this->kernelLumapack, this->layoutLumapack);
    this->pipelineGaussInput = runtime.createComputePipeline(this->kernelGaussInput, this->layoutGaussInput);
}

cv::Mat IQM::GPU::SSIM::computeMetric(const VulkanRuntime &runtime) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineLumapack);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutLumapack, 0, {this->descSetLumapack}, {});

    //shaders work in 16x16 tiles
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

    runtime._cmd_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, imageMemoryBarrier
    );

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGaussInput);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGaussInput, 0, {this->descSetGaussInput}, {});

    std::array valuesGauss = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->sigma)
    };
    runtime._cmd_buffer.pushConstants<int>(this->layoutGaussInput, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

    runtime._cmd_buffer.dispatch(groupsX, groupsY, 1);

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

    runtime._cmd_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {}, {}, gaussImageMemoryBarrier
    );

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    std::array values = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->k_1),
        *reinterpret_cast<int *>(&this->k_2),
        *reinterpret_cast<int *>(&this->sigma)
    };
    runtime._cmd_buffer.pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, values);

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

    const auto size = this->imageParameters.height * this->imageParameters.width * sizeof(float);
    auto [stgBuf, stgMem] = runtime.createBuffer(
        size,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
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
    runtime._cmd_buffer.copyImageToBuffer(this->imageOut->image,  vk::ImageLayout::eGeneral, stgBuf, copyRegion);

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
    dummy.create(static_cast<int>(this->imageParameters.height), static_cast<int>(this->imageParameters.width), CV_32FC1);
    void * outBufData = stgMem.mapMemory(0, this->imageParameters.height * this->imageParameters.width * sizeof(float), {});
    memcpy(dummy.data, outBufData, this->imageParameters.height * this->imageParameters.width * sizeof(float));
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

    void * inBufData = stgMem.mapMemory(0, this->imageParameters.height * this->imageParameters.width * 4, {});
    memcpy(inBufData, image.data, this->imageParameters.height * this->imageParameters.width * 4);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, this->imageParameters.height * this->imageParameters.width * 4, {});
    memcpy(inBufData, ref.data, this->imageParameters.height * this->imageParameters.width * 4);
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
    runtime._cmd_buffer.begin(beginInfo);

    runtime.setImageLayout(this->imageInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageOut->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageLuma->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(this->imageLumaBlurred->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = this->imageParameters.width,
        .bufferImageHeight = this->imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{this->imageParameters.width, this->imageParameters.height, 1}
    };
    runtime._cmd_buffer.copyBufferToImage(stgBuf, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_buffer.copyBufferToImage(stgRefBuf, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

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

    auto imageInfos = createImageInfos({
        this->imageLuma,
        this->imageLumaBlurred,
        this->imageOut,
    });
    
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

    auto lumapackImageInfos = createImageInfos({
        this->imageInput,
        this->imageRef,
        this->imageLuma,
    });

    vk::WriteDescriptorSet writeSetLumapack{
        .dstSet = this->descSetLumapack,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 3,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = lumapackImageInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    auto gaussInputImageInfos = createImageInfos({
        this->imageLuma,
        this->imageLumaBlurred,
    });

    vk::WriteDescriptorSet writeSetGauss{
        .dstSet = this->descSetGaussInput,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = gaussInputImageInfos.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    runtime._device.updateDescriptorSets({writeSet, writeSetLumapack, writeSetGauss}, nullptr);
}

std::vector<vk::DescriptorImageInfo> IQM::GPU::createImageInfos(const std::vector<std::shared_ptr<VulkanImage>> &images) {
    std::vector<vk::DescriptorImageInfo> vec(images.size());

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = vk::DescriptorImageInfo {
            .sampler = nullptr,
            .imageView = images[i]->imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
    }

    return vec;
}

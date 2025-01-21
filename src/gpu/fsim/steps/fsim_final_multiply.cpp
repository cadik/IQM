/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim_final_multiply.h"

IQM::GPU::FSIMFinalMultiply::FSIMFinalMultiply(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/fsim_final_multiply.spv");
    this->sumKernel = runtime.createShaderModule("../shaders_out/fsim_final_sum.spv");

    //custom layout for this pass
    this->descSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 3},
    }));

    this->sumDescSetLayout = std::move(runtime.createDescLayout({
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    const std::vector layouts = {
        *this->descSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    // 1x int - buffer size
    const auto sumRanges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layout = runtime.createPipelineLayout({this->descSetLayout}, {});
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);

    this->sumLayout = runtime.createPipelineLayout({this->sumDescSetLayout}, {sumRanges});
    this->sumPipeline = runtime.createComputePipeline(this->sumKernel, this->sumLayout);

    this->images = std::vector<std::shared_ptr<VulkanImage>>(3);
}

std::pair<float, float> IQM::GPU::FSIMFinalMultiply::computeMetrics(
    const VulkanRuntime &runtime,
    const std::vector<std::shared_ptr<VulkanImage>>& inputImgs,
    const std::vector<std::shared_ptr<VulkanImage>>& gradientImgs,
    const std::vector<std::shared_ptr<VulkanImage>>& pcImgs,
    int width,
    int height
    ) {
    this->prepareImageStorage(runtime, inputImgs, gradientImgs, pcImgs, width, height);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    VulkanRuntime::initImages(runtime._cmd_buffer, this->images);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime.waitForFence(fence);

    return this->sumImages(runtime, width, height);
}

void IQM::GPU::FSIMFinalMultiply::prepareImageStorage(
    const VulkanRuntime &runtime,
    const std::vector<std::shared_ptr<VulkanImage>>& inputImgs,
    const std::vector<std::shared_ptr<VulkanImage>>& gradientImgs,
    const std::vector<std::shared_ptr<VulkanImage>>& pcImgs,
    int width,
    int height
    ) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = vk::Extent3D(width, height, 1),
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

    for (unsigned i = 0; i < 3; i++) {
        this->images[i] = std::move(std::make_shared<VulkanImage>(runtime.createImage(imageInfo)));
    }

    auto inImageInfos = VulkanRuntime::createImageInfos(inputImgs);
    auto gradImageInfos = VulkanRuntime::createImageInfos(gradientImgs);
    auto pcImageInfos = VulkanRuntime::createImageInfos(pcImgs);
    auto outImageInfos = VulkanRuntime::createImageInfos(this->images);

    const auto writeSetIn = VulkanRuntime::createWriteSet(
        this->descSet,
        0,
        inImageInfos
    );

    const auto writeSetGrad = VulkanRuntime::createWriteSet(
        this->descSet,
        1,
        gradImageInfos
    );

    const auto writeSetPc = VulkanRuntime::createWriteSet(
        this->descSet,
        2,
        pcImageInfos
    );

    const auto writeSetOut = VulkanRuntime::createWriteSet(
        this->descSet,
        3,
        outImageInfos
    );

    auto [sumBuf, sumMem] = runtime.createBuffer(
        width * height * sizeof(float),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    sumBuf.bindMemory(sumMem, 0);
    this->sumBuffer = std::move(sumBuf);
    this->sumMemory = std::move(sumMem);

    auto bufInfo = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = this->sumBuffer,
            .offset = 0,
            .range = width * height * sizeof(float),
        }
    };

    const auto writeSetSum = VulkanRuntime::createWriteSet(
        this->sumDescSet,
        0,
        bufInfo
    );

    const std::vector writes = {
        writeSetIn, writeSetGrad, writeSetPc, writeSetOut, writeSetSum
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

std::pair<float, float> IQM::GPU::FSIMFinalMultiply::sumImages(const VulkanRuntime &runtime, int width, int height) {
    auto [stgBuf, stgMem] = runtime.createBuffer(
        3 * sizeof(float),
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);
    auto * bufData = static_cast<float*>(stgMem.mapMemory(0,  3 * sizeof(float), {}));

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint32_t bufferSize = width * height;
    for (unsigned i = 0; i < 3; i++) {
        uint64_t groups = (bufferSize / 128) + 1;
        uint32_t size = bufferSize;

        const vk::BufferImageCopy regionTo {
            .bufferOffset = 0,
            .bufferRowLength = static_cast<unsigned>(width),
            .bufferImageHeight =  static_cast<unsigned>(height),
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{static_cast<unsigned>(width), static_cast<unsigned>(height), 1}
        };

        runtime._cmd_buffer->copyImageToBuffer(this->images[i]->image, vk::ImageLayout::eGeneral, this->sumBuffer, {regionTo});

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = this->sumBuffer,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );

        for (;;) {
            runtime._cmd_buffer->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            runtime._cmd_buffer->dispatch(groups, 1, 1);

            vk::BufferMemoryBarrier barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead,
                .buffer = this->sumBuffer,
                .offset = 0,
                .size = bufferSize * sizeof(float),
            };
            runtime._cmd_buffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlagBits::eDeviceGroup,
                {},
                {barrier},
                {}
            );
            if (groups == 1) {
                break;
            }
            size = groups;
            groups = (groups / 128) + 1;
        }

        const vk::BufferCopy regionFrom = {
            .srcOffset = 0,
            .dstOffset = i * sizeof(float),
            .size = sizeof(float),
        };

        runtime._cmd_buffer->copyBuffer(this->sumBuffer, stgBuf, {regionFrom});

        barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .buffer = this->sumBuffer,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        runtime._cmd_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );
    }

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime.waitForFence(fence);

    float pcm = bufData[0];
    float sim = bufData[1];
    float simc = bufData[2];

    stgMem.unmapMemory();

    return {sim / pcm, simc / pcm};
}

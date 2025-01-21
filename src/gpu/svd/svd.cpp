/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#include "svd.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <execution>

static uint32_t src[] =
#include <svd/svd.inc>
;

IQM::GPU::SVD::SVD(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule(src, sizeof(src));

    const std::vector layouts = {
        *runtime._descLayoutBuffer
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts.data()
    };

    this->descSet = std::move(vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo}.front());

    // 1x int - buffer size
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1);

    this->layout = runtime.createPipelineLayout(layouts, ranges);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

IQM::GPU::SVDResult IQM::GPU::SVD::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    SVDResult res;

    auto bufSize = 2 * 8 * (image.width / 8) * (image.height / 8);
    auto outBufSize = (image.width / 8) * (image.height / 8);
    std::vector<float> data(bufSize);

    this->prepareBuffers(runtime, bufSize * sizeof(float), outBufSize * sizeof(float));

    res.timestamps.mark("buffers prepared");

    cv::Mat inputColor(image.data);
    inputColor = inputColor.reshape(4, image.height);
    cv::Mat refColor(ref.data);
    refColor = refColor.reshape(4, image.height);

    cv::Mat greyInput;
    cv::cvtColor(inputColor, greyInput, cv::COLOR_RGB2GRAY);
    cv::Mat greyRef;
    cv::cvtColor(refColor, greyRef, cv::COLOR_RGB2GRAY);
    cv::Mat inputFloat;
    greyInput.convertTo(inputFloat, CV_32F);
    cv::Mat refFloat;
    greyRef.convertTo(refFloat, CV_32F);

    res.timestamps.mark("image converted");

    // create parallel iterator
    std::vector<int> nums;
    for (int y = 0; (y + 8) < image.height; y+=8) {
        nums.push_back(y);
    }

    std::for_each(std::execution::par, nums.begin(), nums.end(), [&](const int &y) {
        // only process full 8x8 blocks
        for (int x = 0; (x + 8) < image.width; x+=8) {
            cv::Rect crop(x, y, 8, 8);
            cv::Mat srcCrop = inputFloat(crop);
            cv::Mat refCrop = refFloat(crop);

            auto srcSvd = cv::SVD(srcCrop, cv::SVD::NO_UV).w;
            auto refSvd = cv::SVD(refCrop, cv::SVD::NO_UV).w;

            auto start = ((y / 8) * (image.width / 8) + (x / 8)) * 2;

            memcpy(data.data() + (start) * 8, srcSvd.data, 8 * sizeof(float));
            memcpy(data.data() + (start + 1) * 8, refSvd.data, 8 * sizeof(float));
        }
    });

    res.timestamps.mark("end SVD compute");

    void * inBufData = this->stgMemory.mapMemory(0, bufSize * sizeof(float), {});
    memcpy(inBufData, data.data(), bufSize * sizeof(float));
    this->stgMemory.unmapMemory();

    this->copyToGpu(runtime, bufSize * sizeof(float), outBufSize * sizeof(float));

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});
    runtime._cmd_buffer->pushConstants<int>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, bufSize);

    // group takes 128 values, reduces to 8 values
    auto groupsX = (bufSize / 128) + 1;
    runtime._cmd_buffer->dispatch(groupsX, 1, 1);

    runtime._cmd_buffer->end();

    const std::vector cmdBufs = {
        &**runtime._cmd_buffer
    };

    auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
    const vk::SubmitInfo submitInfo{
        .pWaitDstStageMask = &mask,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufs.data()
    };

    const vk::raii::Fence fence{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfo, *fence);
    runtime._device.waitIdle();

    res.timestamps.mark("GPU sum computed");

    this->copyFromGpu(runtime, outBufSize * sizeof(float));

    cv::Mat dummy;
    dummy.create(image.height / 8, image.width / 8, CV_32F);
    void * outBufData = this->stgMemory.mapMemory(0, outBufSize * sizeof(float), {});
    memcpy(dummy.data, outBufData, outBufSize * sizeof(float));
    this->stgMemory.unmapMemory();

    res.timestamps.mark("end GPU writeback");

    std::vector<float> blocks((image.height / 8) * (image.width / 8));
    memcpy(blocks.data(), dummy.data, outBufSize * sizeof(float));
    std::ranges::sort(blocks);

    float middlePoint = blocks[blocks.size() / 2];

    float sum = 0.0;
    for (float block : blocks) {
        sum += std::abs(block - middlePoint);
    }

    res.msvd = sum / static_cast<float>(blocks.size());

    // remap range of output image
    double min, max;
    cv::minMaxLoc(dummy, &min, &max);
    dummy.forEach<float>([max](float& i, const int []) {i = i / static_cast<float>(max);});

    res.height = dummy.rows;
    res.width = dummy.cols;
    res.imageData = std::vector<float>(dummy.rows * dummy.cols);
    memcpy(res.imageData.data(), dummy.data, dummy.rows * dummy.cols * sizeof(float));

    return res;
}

void IQM::GPU::SVD::prepareBuffers(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput) {
    assert(sizeInput > sizeOutput);

    // one staging buffer should be enough
    auto [stgBuf, stgMem] = runtime.createBuffer(
        sizeInput,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    stgBuf.bindMemory(stgMem, 0);
    this->stgBuffer = std::move(stgBuf);
    this->stgMemory = std::move(stgMem);

    auto [buf, mem] = runtime.createBuffer(
        sizeInput,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    buf.bindMemory(mem, 0);
    this->inputBuffer = std::move(buf);
    this->inputMemory = std::move(mem);

    auto [outBuf, outMem] = runtime.createBuffer(
        sizeOutput,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    outBuf.bindMemory(outMem, 0);
    this->outBuffer = std::move(outBuf);
    this->outMemory = std::move(outMem);
}

void IQM::GPU::SVD::copyToGpu(const VulkanRuntime &runtime, size_t sizeInput, size_t sizeOutput) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeInput,
    };
    runtime._cmd_buffer->copyBuffer(this->stgBuffer, this->inputBuffer, copyRegion);

    runtime._cmd_buffer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eAllCommands};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfoCopy, *fenceCopy);
    runtime._device.waitIdle();

    std::vector bufInfos = {
        vk::DescriptorBufferInfo {
            .buffer = this->inputBuffer,
            .offset = 0,
            .range = sizeInput,
        },
        vk::DescriptorBufferInfo {
            .buffer = this->outBuffer,
            .offset = 0,
            .range = sizeOutput,
        },
    };

    vk::WriteDescriptorSet writeSet{
        .dstSet = this->descSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufInfos.data(),
        .pTexelBufferView = nullptr,
    };

    runtime._device.updateDescriptorSets(writeSet, nullptr);
}

void IQM::GPU::SVD::copyFromGpu(const VulkanRuntime &runtime, size_t sizeOutput) {
    runtime._cmd_buffer->reset();
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfoCopy);

    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeOutput,
    };
    runtime._cmd_buffer->copyBuffer(this->outBuffer, this->stgBuffer, copyRegion);

    runtime._cmd_buffer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_buffer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._queue->submit(submitInfoCopy, *fenceCopy);
    runtime.waitForFence(fenceCopy);
}

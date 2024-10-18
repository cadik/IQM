#include "ssim.h"

IQM::GPU::SSIM::SSIM(const VulkanRuntime &runtime) {
    this->kernel = runtime.createShaderModule("../shaders_out/ssim.spv");

    vk::PipelineLayoutCreateInfo layoutInfo = {};

    this->layout = runtime.createPipelineLayout(layoutInfo);
    this->pipeline = runtime.createComputePipeline(this->kernel, this->layout);
}

void IQM::GPU::SSIM::computeMetric(const VulkanRuntime &runtime) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer.begin(beginInfo);

    runtime._cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    runtime._cmd_buffer.dispatch(1000, 1000, 1000);

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


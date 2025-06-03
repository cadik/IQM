/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/psnr.h>

static std::vector<uint32_t> srcPack =
#include <psnr/pack.inc>
;

static std::vector<uint32_t> srcSum =
#include <psnr/sum.inc>
;

static std::vector<uint32_t> srcPost =
#include <psnr/postprocess.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::PSNR::PSNR(const vk::raii::Device &device) {
    const auto smPack = VulkanRuntime::createShaderModule(device, srcPack);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);
    const auto smPost = VulkanRuntime::createShaderModule(device, srcPost);

    this->descPool = VulkanRuntime::createDescPool(device, 4, {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 16},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 16}
    });

    this->descSetLayoutPack = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->descSetLayoutSum = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allocateLayouts = {
        *this->descSetLayoutPack,
        *this->descSetLayoutSum,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allocateLayouts.size()),
        .pSetLayouts = allocateLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetPack = std::move(sets[0]);
    this->descSetSum = std::move(sets[1]);

    // 1x int - data size
    const auto rangeSum = VulkanRuntime::createPushConstantRange( sizeof(int));

    this->layoutPack = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutPack}, rangeSum);
    this->layoutSum = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutSum}, rangeSum);

    this->pipelinePack = VulkanRuntime::createComputePipeline(device, smPack, this->layoutPack);
    this->pipelineSum = VulkanRuntime::createComputePipeline(device, smSum, this->layoutSum);
    this->pipelinePost = VulkanRuntime::createComputePipeline(device, smPost, this->layoutSum);
}

void IQM::PSNR::computeMetric(const PSNRInput &input) {
    this->initDescriptors(input);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelinePack);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutPack, 0, {this->descSetPack}, {});
    input.cmdBuf->pushConstants<int>(this->layoutPack, vk::ShaderStageFlagBits::eCompute, 0, static_cast<int>(input.variant));

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    if (input.imgOut != nullptr) {
        auto region = vk::BufferImageCopy {
            .bufferOffset = 0,
            .bufferRowLength = input.width,
            .bufferImageHeight = input.height,
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{input.width, input.height, 1},
        };

        input.cmdBuf->copyBufferToImage(*input.bufSum, *input.imgOut, vk::ImageLayout::eGeneral, {region});

        memoryBarrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
        };

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
        );
    }

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSum);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSum, 0, {this->descSetSum}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = (input.width) * (input.height);
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;

    for (;;) {
        input.cmdBuf->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, 0, size);
        input.cmdBuf->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = *input.bufSum,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );
        if (groups == 1) {
            break;
        }
        size = groups;
        groups = (groups / sumSize) + 1;
    }

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelinePost);
    input.cmdBuf->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, 0, bufferSize);

    input.cmdBuf->dispatch(1, 1, 1);

    memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::PSNR::initDescriptors(const PSNRInput &input) {
    const unsigned size = input.width * input.height * sizeof(float);

    const auto inputImageInfos = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    std::vector bufInfos = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufSum,
            .offset = 0,
            .range = size,
        }
    };

    // input
    auto writeSetPackIn = VulkanRuntime::createWriteSet(
        this->descSetPack,
        0,
        inputImageInfos
    );

    auto writeSetPackOut = VulkanRuntime::createWriteSet(
        this->descSetPack,
        1,
        bufInfos
    );

    // sum
    auto writeSetSum = VulkanRuntime::createWriteSet(
        this->descSetSum,
        0,
        bufInfos
    );

    input.device->updateDescriptorSets({writeSetPackIn, writeSetPackOut, writeSetSum}, nullptr);
}

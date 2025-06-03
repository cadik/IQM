/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/ssim.h>

static std::vector<uint32_t> src =
#include <ssim/ssim.inc>
;

static std::vector<uint32_t> srcLumapack =
#include <ssim/ssim_lumapack.inc>
;

static std::vector<uint32_t> srcGaussHorizontal =
#include <ssim/ssim_gauss_horizontal.inc>
;

static std::vector<uint32_t> srcGauss =
#include <ssim/ssim_gauss.inc>
;

static std::vector<uint32_t> srcMssim =
#include <ssim/mssim_sum.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::SSIM::SSIM(const vk::raii::Device &device) {
    const auto smSsim = VulkanRuntime::createShaderModule(device, src);
    const auto smLumapack = VulkanRuntime::createShaderModule(device,srcLumapack);
    const auto smGaussHorizontal = VulkanRuntime::createShaderModule(device, srcGaussHorizontal);
    const auto smGauss = VulkanRuntime::createShaderModule(device, srcGauss);
    const auto smMssim = VulkanRuntime::createShaderModule(device, srcMssim);

    this->descPool = VulkanRuntime::createDescPool(device, 4, {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 20},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 20}
    });

    this->descSetLayoutLumapack = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 5},
    });

    this->descSetLayoutSsim = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 5},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->descSetLayoutMssim = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allocateLayouts = {
        *this->descSetLayoutLumapack,
        *this->descSetLayoutSsim,
        *this->descSetLayoutSsim,
        *this->descSetLayoutMssim
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allocateLayouts.size()),
        .pSetLayouts = allocateLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetLumapack = std::move(sets[0]);
    this->descSetGauss = std::move(sets[1]);
    this->descSetSsim = std::move(sets[2]);
    this->descSetMssim = std::move(sets[3]);

    // 1x int - kernel size
    // 3x float - K_1, K_2, sigma
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1 + sizeof(float) * 3);

    // 2x int - kernel size, kernel direction
    // 1x float - sigma
    const auto rangesGauss = VulkanRuntime::createPushConstantRange(2 * sizeof(int) + sizeof(float));

    // 1x int - data size
    const auto rangeMssim = VulkanRuntime::createPushConstantRange( sizeof(int));

    this->layoutLumapack = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutLumapack}, {});
    this->layoutGauss = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutSsim}, rangesGauss);
    this->layoutSsim = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutSsim}, ranges);
    this->layoutMssim = VulkanRuntime::createPipelineLayout(device, {*this->descSetLayoutMssim}, rangeMssim);

    this->pipelineLumapack = VulkanRuntime::createComputePipeline(device, smLumapack, this->layoutLumapack);
    this->pipelineGauss = VulkanRuntime::createComputePipeline(device, smGauss, this->layoutGauss);
    this->pipelineGaussHorizontal = VulkanRuntime::createComputePipeline(device, smGaussHorizontal, this->layoutGauss);
    this->pipelineSsim = VulkanRuntime::createComputePipeline(device, smSsim, this->layoutSsim);
    this->pipelineMssim = VulkanRuntime::createComputePipeline(device, smMssim, this->layoutMssim);
}

void IQM::SSIM::computeMetric(const SSIMInput &input) {
    this->initDescriptors(input);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineLumapack);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutLumapack, 0, {this->descSetLumapack}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    for (int i = 0; i < 5; i++) {
        input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGaussHorizontal);
        input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGauss, 0, {this->descSetGauss}, {});

        std::array valuesGauss = {
            this->kernelSize,
            *reinterpret_cast<int *>(&this->sigma),
            i,
        };
        input.cmdBuf->pushConstants<int>(this->layoutGauss, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

        input.cmdBuf->dispatch(groupsX, groupsY, 1);

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
        );

        valuesGauss = {
            this->kernelSize,
            *reinterpret_cast<int *>(&this->sigma),
            i,
        };

        input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGauss);
        input.cmdBuf->pushConstants<int>(this->layoutGauss, vk::ShaderStageFlagBits::eCompute, 0, valuesGauss);

        input.cmdBuf->dispatch(groupsX, groupsY, 1);

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
        );
    }

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSsim);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSsim, 0, {this->descSetSsim}, {});

    std::array values = {
        this->kernelSize,
        *reinterpret_cast<int *>(&this->k_1),
        *reinterpret_cast<int *>(&this->k_2),
        *reinterpret_cast<int *>(&this->sigma)
    };
    input.cmdBuf->pushConstants<int>(this->layoutSsim, vk::ShaderStageFlagBits::eCompute, 0, values);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    const auto halfOffset = (this->kernelSize - 1) / 2;
    const auto offset = this->kernelSize - 1;

    vk::BufferImageCopy copyMssimRegion{
        .bufferOffset = 0,
        .bufferRowLength = input.width - offset,
        .bufferImageHeight = input.height - offset,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{halfOffset, halfOffset, 0},
        .imageExtent = vk::Extent3D{input.width - offset, input.height - offset, 1}
    };
    input.cmdBuf->copyImageToBuffer(*input.imgOut,  vk::ImageLayout::eGeneral, *input.bufMssim, copyMssimRegion);

    memBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    memBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {memBarrier},
        {},
        {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineMssim);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutMssim, 0, {this->descSetMssim}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = (input.width - offset) * (input.height - offset);
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;

    for (;;) {
        input.cmdBuf->pushConstants<unsigned>(this->layoutMssim, vk::ShaderStageFlagBits::eCompute, 0, size);
        input.cmdBuf->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = *input.bufMssim,
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
}

void IQM::SSIM::initDescriptors(const SSIMInput &input) {
    const unsigned sizeTrimmed = (input.width - this->kernelSize + 1) * (input.height - this->kernelSize + 1) * sizeof(float);

    const auto imageInfosIntermediate = VulkanRuntime::createImageInfos({
        input.ivMeanTest,
        input.ivMeanRef,
        input.ivVarTest,
        input.ivVarRef,
        input.ivCovar,
    });
    const auto outImageInfos = VulkanRuntime::createImageInfos({input.ivOut});

    const auto inputImageInfos = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    std::vector bufInfos = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufMssim,
            .offset = 0,
            .range = sizeTrimmed,
        }
    };

    // input
    auto writeSetLumapackIn = VulkanRuntime::createWriteSet(
        this->descSetLumapack,
        0,
        inputImageInfos
    );

    auto writeSetLumapackOut = VulkanRuntime::createWriteSet(
        this->descSetLumapack,
        1,
        imageInfosIntermediate
    );

    // gauss
    auto writeSetGaussFlip = VulkanRuntime::createWriteSet(
        this->descSetGauss,
        0,
        imageInfosIntermediate
    );

    auto writeSetGaussFlop = VulkanRuntime::createWriteSet(
        this->descSetGauss,
        1,
        outImageInfos
    );

    // ssim
    auto writeSetSsimIn = VulkanRuntime::createWriteSet(
        this->descSetSsim,
        0,
        imageInfosIntermediate
    );

    auto writeSetSsimOut = VulkanRuntime::createWriteSet(
        this->descSetSsim,
        1,
        outImageInfos
    );

    // mssim
    auto writeSetSum = VulkanRuntime::createWriteSet(
        this->descSetMssim,
        0,
        bufInfos
    );

    input.device->updateDescriptorSets({writeSetLumapackIn, writeSetLumapackOut, writeSetGaussFlip, writeSetGaussFlop, writeSetSsimIn, writeSetSsimOut, writeSetSum}, nullptr);
}

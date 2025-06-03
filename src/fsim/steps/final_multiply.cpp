/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/final_multiply.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_final_multiply.inc>
;

static std::vector<uint32_t> srcSum =
#include <fsim/fsim_final_sum.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMFinalMultiply::FSIMFinalMultiply(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smMul = VulkanRuntime::createShaderModule(device, src);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 3},
    });

    this->sumDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector layouts = {
        *this->descSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(sets[0]);
    this->sumDescSet = std::move(sets[1]);

    // 1x int - buffer size
    const auto sumRanges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {});
    this->pipeline = VulkanRuntime::createComputePipeline(device, smMul, this->layout);

    this->sumLayout = VulkanRuntime::createPipelineLayout(device, {this->sumDescSetLayout}, {sumRanges});
    this->sumPipeline = VulkanRuntime::createComputePipeline(device, smSum, this->sumLayout);
}

void IQM::FSIMFinalMultiply::computeMetrics(const FSIMInput &input, unsigned width, unsigned height) {
    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    this->sumImages(input, width, height);
}

void IQM::FSIMFinalMultiply::setUpDescriptors(const FSIMInput &input, const unsigned width, const unsigned height) {
    auto inImageInfos = VulkanRuntime::createImageInfos({input.ivTestDown, input.ivRefDown});
    auto gradImageInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[0], input.ivTempFloat[1]});
    auto pcImageInfos = VulkanRuntime::createImageInfos({input.ivTempFloat[2], input.ivTempFloat[3]});
    auto outImageInfos = VulkanRuntime::createImageInfos(std::vector(std::begin(input.ivFinalSums), std::end(input.ivFinalSums)));

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

    // reuse iFFT buffer, since it's big enough
    auto bufInfo = std::vector{
        vk::DescriptorBufferInfo {
            .buffer = **input.bufIfft,
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

    input.device->updateDescriptorSets(writes, nullptr);
}

void IQM::FSIMFinalMultiply::sumImages(const FSIMInput &input, const unsigned width, const unsigned height) {
    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    uint32_t bufferSize = width * height;
    for (unsigned i = 0; i < 3; i++) {
        uint64_t groups = (bufferSize / 1024) + 1;
        uint32_t size = bufferSize;

        const vk::BufferImageCopy regionTo {
            .bufferOffset = 0,
            .bufferRowLength = static_cast<unsigned>(width),
            .bufferImageHeight =  static_cast<unsigned>(height),
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{static_cast<unsigned>(width), static_cast<unsigned>(height), 1}
        };

        input.cmdBuf->copyImageToBuffer(*input.imgFinalSums[i], vk::ImageLayout::eGeneral, *input.bufIfft, {regionTo});

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = **input.bufIfft,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );

        for (;;) {
            input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
            input.cmdBuf->dispatch(groups, 1, 1);

            vk::BufferMemoryBarrier barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead,
                .buffer = **input.bufIfft,
                .offset = 0,
                .size = bufferSize * sizeof(float),
            };
            input.cmdBuf->pipelineBarrier(
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
            groups = (groups / 1024) + 1;
        }

        const vk::BufferCopy regionFrom = {
            .srcOffset = 0,
            .dstOffset = i * sizeof(float),
            .size = sizeof(float),
        };

        input.cmdBuf->copyBuffer(*input.bufIfft, *input.bufFft, {regionFrom});

        barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .buffer = *input.bufIfft,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );
    }
}

/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <cmath>
#include <IQM/flip.h>

static std::vector<uint32_t> srcInputConvert =
#include <flip/srgb_to_ycxcz.inc>
;

static std::vector<uint32_t> srcFeatureFilterCreate =
#include <flip/feature_filter.inc>
;

static std::vector<uint32_t> srcFeatureFilterNormalize =
#include <flip/feature_filter_normalize.inc>
;

static std::vector<uint32_t> srcFeatureFilterHorizontal =
#include <flip/feature_filter_horizontal.inc>
;

static std::vector<uint32_t> srcFeatureDetect =
#include <flip/feature_detection.inc>
;

static std::vector<uint32_t> srcErrCombine =
#include <flip/combine_error_maps.inc>
;

static std::vector<uint32_t> srcSum =
#include <flip/sum.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FLIP::FLIP(const vk::raii::Device &device):
descPool(VulkanRuntime::createDescPool(device, 64, {
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 128},
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 32}
})),
colorPipeline(device, descPool)
{
    const auto smInputConvert = VulkanRuntime::createShaderModule(device, srcInputConvert);
    const auto smFeatureFilterCreate = VulkanRuntime::createShaderModule(device, srcFeatureFilterCreate);
    const auto smFeatureFilterNormalize = VulkanRuntime::createShaderModule(device, srcFeatureFilterNormalize);
    const auto smFeatureFilterHorizontal = VulkanRuntime::createShaderModule(device, srcFeatureFilterHorizontal);
    const auto smFeatureDetect = VulkanRuntime::createShaderModule(device, srcFeatureDetect);
    const auto smErrorCombine = VulkanRuntime::createShaderModule(device, srcErrCombine);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);

    this->inputConvertDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 2},
    });

    this->featureFilterCreateDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->featureFilterHorizontalDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->featureDetectDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageImage, 1},
    });

    this->errorCombineDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->sumDescSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allDescLayouts = {
        *this->inputConvertDescSetLayout,
        *this->featureFilterCreateDescSetLayout,
        *this->featureFilterHorizontalDescSetLayout,
        *this->featureDetectDescSetLayout,
        *this->errorCombineDescSetLayout,
        *this->sumDescSetLayout,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allDescLayouts.size()),
        .pSetLayouts = allDescLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->inputConvertDescSet = std::move(sets[0]);
    this->featureFilterCreateDescSet = std::move(sets[1]);
    this->featureFilterHorizontalDescSet = std::move(sets[2]);
    this->featureDetectDescSet = std::move(sets[3]);
    this->errorCombineDescSet = std::move(sets[4]);
    this->sumDescSet = std::move(sets[5]);

    this->inputConvertLayout = VulkanRuntime::createPipelineLayout(device, {this->inputConvertDescSetLayout}, {});
    this->inputConvertPipeline = VulkanRuntime::createComputePipeline(device, smInputConvert, this->inputConvertLayout);

    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(float));
    this->featureFilterCreateLayout = VulkanRuntime::createPipelineLayout(device, {this->featureFilterCreateDescSetLayout}, ranges);
    this->featureFilterCreatePipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterCreate, this->featureFilterCreateLayout);
    this->featureFilterNormalizePipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterNormalize, this->featureFilterCreateLayout);

    const auto rangesHor = VulkanRuntime::createPushConstantRange(3 * sizeof(unsigned));
    this->featureFilterHorizontalLayout = VulkanRuntime::createPipelineLayout(device, {this->featureFilterHorizontalDescSetLayout}, rangesHor);
    this->featureFilterHorizontalPipeline = VulkanRuntime::createComputePipeline(device, smFeatureFilterHorizontal, this->featureFilterHorizontalLayout);

    this->featureDetectLayout = VulkanRuntime::createPipelineLayout(device, {this->featureDetectDescSetLayout}, rangesHor);
    this->featureDetectPipeline = VulkanRuntime::createComputePipeline(device, smFeatureDetect, this->featureDetectLayout);

    this->errorCombineLayout = VulkanRuntime::createPipelineLayout(device, {this->errorCombineDescSetLayout}, ranges);
    this->errorCombinePipeline = VulkanRuntime::createComputePipeline(device, smErrorCombine, this->errorCombineLayout);

    this->sumLayout = VulkanRuntime::createPipelineLayout(device, {this->sumDescSetLayout}, ranges);
    this->sumPipeline = VulkanRuntime::createComputePipeline(device, smSum, this->sumLayout);
}

void IQM::FLIP::computeMetric(const FLIPInput &input) {
    float pixelsPerDegree = FLIP::pixelsPerDegree(input.args);

    this->setUpDescriptors(input);
    this->colorPipeline.setUpDescriptors(input);
    this->convertToYCxCz(input);
    this->createFeatureFilters(input);
    this->computeFeatureErrorMap(input);
    this->colorPipeline.prefilter(input, pixelsPerDegree);
    this->colorPipeline.computeErrorMap(input);
    this->computeFinalErrorMap(input);
    this->computeMean(input);
}

float IQM::FLIP::pixelsPerDegree(const FLIPArguments &args) {
    return args.monitor_distance * (args.monitor_resolution_x / args.monitor_width) * (std::numbers::pi / 180.0);
}

unsigned IQM::FLIP::spatialKernelSize(const FLIPArguments &args) {
    return 2 * static_cast<int>(std::ceil(3 * std::sqrt(0.04 / (2.0 * std::pow(std::numbers::pi, 2.0))) * pixelsPerDegree(args))) + 1;
}

unsigned IQM::FLIP::featureKernelSize(const FLIPArguments &args) {
    return 2 * static_cast<int>(std::ceil(3 * 0.5 * 0.082 * pixelsPerDegree(args))) + 1;
}

void IQM::FLIP::convertToYCxCz(const FLIPInput& input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->inputConvertPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->inputConvertLayout, 0, {this->inputConvertDescSet}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);
}

void IQM::FLIP::createFeatureFilters(const FLIPInput& input) {
    float pixelsPerDegree = FLIP::pixelsPerDegree(input.args);
    int gaussianKernelSize = FLIP::featureKernelSize(input.args);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterCreatePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixelsPerDegree);

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(gaussianKernelSize, gaussianKernelSize, 16);

    input.cmdBuf->dispatch(groupsX, 1, 2);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterNormalizePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterCreateLayout, 0, {this->featureFilterCreateDescSet}, {});
    input.cmdBuf->pushConstants<float>(this->featureFilterCreateLayout, vk::ShaderStageFlagBits::eCompute, 0, pixelsPerDegree);

    input.cmdBuf->dispatch(groupsX, 1, 2);
}

void IQM::FLIP::computeFeatureErrorMap(const FLIPInput& input) {
    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureFilterHorizontalPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureFilterHorizontalLayout, 0, {this->featureFilterHorizontalDescSet}, {});
    input.cmdBuf->pushConstants<uint32_t>(this->featureFilterHorizontalLayout, vk::ShaderStageFlagBits::eCompute, 0 * sizeof(float), input.width * input.height);
    input.cmdBuf->pushConstants<uint32_t>(this->featureFilterHorizontalLayout, vk::ShaderStageFlagBits::eCompute, 1 * sizeof(float), input.width);
    input.cmdBuf->pushConstants<uint32_t>(this->featureFilterHorizontalLayout, vk::ShaderStageFlagBits::eCompute, 2 * sizeof(float), input.height);

    auto groups = VulkanRuntime::compute1DGroupCount(input.width * input.height, 1024);

    input.cmdBuf->dispatch(groups, 1, 2);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->featureDetectPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->featureDetectLayout, 0, {this->featureDetectDescSet}, {});
    input.cmdBuf->pushConstants<uint32_t>(this->featureDetectLayout, vk::ShaderStageFlagBits::eCompute, 0 * sizeof(float), input.width * input.height);
    input.cmdBuf->pushConstants<uint32_t>(this->featureDetectLayout, vk::ShaderStageFlagBits::eCompute, 1 * sizeof(float), input.width);
    input.cmdBuf->pushConstants<uint32_t>(this->featureDetectLayout, vk::ShaderStageFlagBits::eCompute, 2 * sizeof(float), input.height);

    input.cmdBuf->dispatch(groups, 1, 1);

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::FLIP::computeFinalErrorMap(const FLIPInput& input) {
    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->errorCombinePipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->errorCombineLayout, 0, {this->errorCombineDescSet}, {});
    input.cmdBuf->pushConstants<uint32_t>(this->errorCombineLayout, vk::ShaderStageFlagBits::eCompute, 0, input.width * input.height);

    auto groups = VulkanRuntime::compute1DGroupCount(input.width * input.height, 1024);

    input.cmdBuf->dispatch(groups, 1, 1);

    memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );

    vk::BufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = input.width,
        .bufferImageHeight = input.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{input.width, input.height, 1}
    };
    input.cmdBuf->copyBufferToImage(*input.buffer, *input.imgOut, vk::ImageLayout::eGeneral, {region});

    memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::FLIP::computeMean(const FLIPInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->sumPipeline);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->sumLayout, 0, {this->sumDescSet}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = (input.width) * (input.height);
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;

    for (;;) {
        input.cmdBuf->pushConstants<unsigned>(this->sumLayout, vk::ShaderStageFlagBits::eCompute, 0, size);
        input.cmdBuf->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = *input.buffer,
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

void IQM::FLIP::setUpDescriptors(const FLIPInput& input) {
    auto rgbRange = input.width * input.height * sizeof(float) * 3;
    auto floatRange = input.width * input.height * sizeof(float);

    auto imageInfos = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    auto yccOutBufInfos = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = 0,
            .range = rgbRange,
        },
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange,
            .range = rgbRange,
        }
    };

    auto featureFilterImageInfos = VulkanRuntime::createImageInfos({
        input.ivFeatFilter,
    });

    auto tempFeatureFilterBufInfos = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange * 2,
            .range = rgbRange,
        },
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange * 3,
            .range = rgbRange,
        }
    };

    auto outFeatureBufInfo = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange * 4,
            .range = floatRange,
        }
    };

    auto outBufInfo = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = 0,
            .range = floatRange,
        }
    };

    auto errorBufInfos = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange * 4,
            .range = floatRange,
        },
        vk::DescriptorBufferInfo {
            .buffer = *input.buffer,
            .offset = rgbRange * 2,
            .range = floatRange,
        }
    };

    auto writeSetConvertInput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        0,
        imageInfos
    );

    auto writeSetConvertOutput = VulkanRuntime::createWriteSet(
        this->inputConvertDescSet,
        1,
        yccOutBufInfos
    );

    auto writeSetFeatureFilter = VulkanRuntime::createWriteSet(
        this->featureFilterCreateDescSet,
        0,
        featureFilterImageInfos
    );

    auto writeSetHorizontalInput = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        0,
        yccOutBufInfos
    );

    auto writeSetHorizontalOutput = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        1,
        tempFeatureFilterBufInfos
    );

    auto writeSetHorizontalFilters = VulkanRuntime::createWriteSet(
        this->featureFilterHorizontalDescSet,
        2,
        featureFilterImageInfos
    );

    auto writeSetDetectInput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        0,
        tempFeatureFilterBufInfos
    );

    auto writeSetDetectOutput = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        1,
        outFeatureBufInfo
    );

    auto writeSetDetectFilters = VulkanRuntime::createWriteSet(
        this->featureDetectDescSet,
        2,
        featureFilterImageInfos
    );

    auto writeSetFinalIn = VulkanRuntime::createWriteSet(
        this->errorCombineDescSet,
        0,
        errorBufInfos
    );

    auto writeSetFinalOut = VulkanRuntime::createWriteSet(
        this->errorCombineDescSet,
        1,
        outBufInfo
    );

    auto writeSetSum = VulkanRuntime::createWriteSet(
        this->sumDescSet,
        0,
        outBufInfo
    );

    input.device->updateDescriptorSets({
        writeSetConvertInput, writeSetConvertOutput,
        writeSetFeatureFilter,
        writeSetHorizontalInput, writeSetHorizontalFilters, writeSetHorizontalOutput,
        writeSetDetectInput, writeSetDetectFilters, writeSetDetectOutput,
        writeSetFinalIn, writeSetFinalOut,
        writeSetSum
    }, nullptr);
}

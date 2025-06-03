/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim.h>

#include <IQM/fsim/fft_planner.h>

static std::vector<uint32_t> srcDownscale =
#include <fsim/fsim_downsample.inc>
;

static std::vector<uint32_t> srcGradient =
#include <fsim/fsim_gradientmap.inc>
;

static std::vector<uint32_t> srcExtractLuma =
#include <fsim/fsim_extractluma.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIM::FSIM(const vk::raii::Device &device):
descPool(VulkanRuntime::createDescPool(device, 64, {
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 128},
    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 32}
})),
logGaborFilter(device, descPool),
angularFilter(device, descPool),
combinations(device, descPool),
sumFilterResponses(device, descPool),
noise_power(device, descPool),
estimateEnergy(device, descPool),
phaseCongruency(device, descPool),
final_multiply(device, descPool)
{
    const auto smDownscale = VulkanRuntime::createShaderModule(device, srcDownscale);
    const auto smGradientMap = VulkanRuntime::createShaderModule(device, srcGradient);
    const auto smExtractLuma = VulkanRuntime::createShaderModule(device, srcExtractLuma);

    this->descSetLayoutImageOp = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
    });

    this->descSetLayoutImBufOp = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector allLayouts = {
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImageOp,
        *this->descSetLayoutImBufOp,
        *this->descSetLayoutImBufOp,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(allLayouts.size()),
        .pSetLayouts = allLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetDownscale = std::move(sets[0]);
    this->descSetGradientMap = std::move(sets[1]);
    this->descSetExtractLumaIn = std::move(sets[2]);
    this->descSetExtractLumaRef = std::move(sets[3]);

    // 1x int - kernel size
    const auto downsampleRanges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layoutDownscale = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImageOp}, downsampleRanges);
    this->pipelineDownscale = VulkanRuntime::createComputePipeline(device, smDownscale, this->layoutDownscale);

    this->layoutGradientMap = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImageOp}, {});
    this->pipelineGradientMap = VulkanRuntime::createComputePipeline(device, smGradientMap, this->layoutGradientMap);

    this->layoutExtractLuma = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutImBufOp}, {});
    this->pipelineExtractLuma = VulkanRuntime::createComputePipeline(device, smExtractLuma, this->layoutExtractLuma);
}

void IQM::FSIM::computeMetric(const FSIMInput &input) {
    const int F = computeDownscaleFactor(input.width, input.height);
    const auto widthDownscale = static_cast<int>(std::round(static_cast<float>(input.width) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<int>(std::round(static_cast<float>(input.height) / static_cast<float>(F)));

    const auto sortSize = (widthDownscale * heightDownscale) * sizeof(float);
    FftBufferPartitions partitions {
        .sort = 0,
        .sortTemp = sortSize,
        .sortHist = 2 * sortSize,
        .noiseLevels = 2 * sortSize + sortBufSize(widthDownscale, heightDownscale),
        .noisePowers = 2 * sortSize + sortBufSize(widthDownscale, heightDownscale) + FSIM_ORIENTATIONS * sizeof(float),
        .end = 2 * sortSize + sortBufSize(widthDownscale, heightDownscale) + FSIM_ORIENTATIONS * sizeof(float) + 2 * FSIM_ORIENTATIONS * sizeof(float),
    };

    this->initDescriptors(input, partitions);

    this->computeDownscaledImages(input, F, widthDownscale, heightDownscale);
    this->logGaborFilter.constructFilter(input, widthDownscale, heightDownscale);
    this->angularFilter.constructFilter(input, widthDownscale, heightDownscale);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        nullptr,
        nullptr
    );

    this->computeFft(input, widthDownscale, heightDownscale);
    this->combinations.combineFilters(input, widthDownscale, heightDownscale, partitions);
    this->computeMassInverseFft(input);
    this->sumFilterResponses.computeSums(input, widthDownscale, heightDownscale);;
    this->noise_power.computeNoisePower(input, widthDownscale, heightDownscale);
    this->estimateEnergy.estimateEnergy(input, widthDownscale, heightDownscale);
    this->createGradientMap(input, widthDownscale, heightDownscale);
    this->phaseCongruency.compute(input, widthDownscale, heightDownscale);
    this->final_multiply.computeMetrics(input, widthDownscale, heightDownscale);
}

std::pair<unsigned, unsigned> IQM::FSIM::downscaledSize(const unsigned width, const unsigned height) {
    const int F = computeDownscaleFactor(width, height);
    const auto widthDownscale = static_cast<unsigned>(std::round(static_cast<float>(width) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<unsigned>(std::round(static_cast<float>(height) / static_cast<float>(F)));

    return std::make_pair(widthDownscale, heightDownscale);
}

unsigned IQM::FSIM::sortBufSize(const unsigned dWidth, const unsigned dHeight) {
    uint32_t sortGlobalInvocationSize = (dWidth * dHeight) / 32;
    uint32_t remainder = (dWidth * dHeight) % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    return nSortWorkgroups * 256 * sizeof(uint32_t);
}

int IQM::FSIM::computeDownscaleFactor(const int width, const int height) {
    auto smallerDim = std::min(width, height);
    return std::max(1, static_cast<int>(std::round(smallerDim / 256.0)));
}

void IQM::FSIM::initDescriptors(const FSIMInput &input, const FftBufferPartitions& partitions) {
    auto [dWidth, dHeight] = downscaledSize(input.width, input.height);

    auto imageInfosInput = VulkanRuntime::createImageInfos({
        input.ivTest,
        input.ivRef,
    });

    auto imageInfosOut = VulkanRuntime::createImageInfos({
        input.ivTestDown,
        input.ivRefDown,
    });

    auto imageInfosGradIn = VulkanRuntime::createImageInfos({
        input.ivTestDown,
        input.ivRefDown,
    });

    auto imageInfosGradOut = VulkanRuntime::createImageInfos({
        input.ivTempFloat[0],
        input.ivTempFloat[1],
    });

    auto writeSetInput = VulkanRuntime::createWriteSet(
        this->descSetDownscale,
        0,
        imageInfosInput
    );

    auto writeSetOut = VulkanRuntime::createWriteSet(
        this->descSetDownscale,
        1,
        imageInfosOut
    );

    auto writeSetGradIn = VulkanRuntime::createWriteSet(
        this->descSetGradientMap,
        0,
        imageInfosGradIn
    );

    auto writeSetGradOut = VulkanRuntime::createWriteSet(
        this->descSetGradientMap,
        1,
        imageInfosGradOut
    );

    // image size * 2 float components (complex numbers) * 2 batches
    uint64_t bufferSize = dWidth * dHeight * sizeof(float) * 2 * 2;
    std::vector bufIn = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufFft,
            .offset = 0,
            .range = bufferSize / 2,
        }
    };
    auto imInfoInImage = VulkanRuntime::createImageInfos({input.ivTestDown});
    auto writeSetInImage = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaIn,
        0,
        imInfoInImage
    );

    auto writeSetInBuf = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaIn,
        1,
        bufIn
    );

    std::vector bufRef = {
        vk::DescriptorBufferInfo{
            .buffer = *input.bufFft,
            .offset = bufferSize / 2,
            .range = bufferSize / 2,
        }
    };

    auto imInfoRefImage = VulkanRuntime::createImageInfos({input.ivRefDown});
    auto writeSetRefImage = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaRef,
        0,
        imInfoRefImage
    );

    auto writeSetRefBuf = VulkanRuntime::createWriteSet(
        this->descSetExtractLumaRef,
        1,
        bufRef
    );

    const std::vector writes = {
        writeSetInput, writeSetOut, writeSetGradIn, writeSetGradOut, writeSetInImage, writeSetInBuf, writeSetRefImage, writeSetRefBuf
    };

    input.device->updateDescriptorSets(writes, nullptr);

    this->angularFilter.setUpDescriptors(input);
    this->estimateEnergy.setUpDescriptors(input, dWidth, dHeight);
    this->logGaborFilter.setUpDescriptors(input);
    this->sumFilterResponses.setUpDescriptors(input, dWidth, dHeight);
    this->final_multiply.setUpDescriptors(input, dWidth, dHeight);
    this->combinations.setUpDescriptors(input, dWidth, dHeight);
    this->noise_power.setUpDescriptors(input, dWidth, dHeight, partitions);
    this->phaseCongruency.setUpDescriptors(input, dWidth, dHeight, partitions);
}

void IQM::FSIM::computeDownscaledImages(const FSIMInput &input, int factor, int width, int height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineDownscale);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscale}, {});

    input.cmdBuf->pushConstants<int>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, factor);

    input.cmdBuf->dispatch(width * height, 1, 2);
}

void IQM::FSIM::createGradientMap(const FSIMInput& input, int width, int height) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGradientMap);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMap}, {});

    //shader works in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 2);
}

void IQM::FSIM::computeFft(const FSIMInput &input, const unsigned width, const unsigned height) {
    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineExtractLuma);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaIn}, {});
    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaRef}, {});
    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {barrier},
        nullptr,
        nullptr
    );

    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **input.cmdBuf;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = **input.bufFft;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(input.fftApplication, -1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append FFT: " + std::to_string(res);
        throw std::runtime_error(err);
    }
}

void IQM::FSIM::computeMassInverseFft(const FSIMInput &input) {
    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **input.cmdBuf;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = **input.bufIfft;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(input.fftApplicationInverse, 1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append inverse FFT: " + std::to_string(res);
        throw std::runtime_error(err);
    }
}

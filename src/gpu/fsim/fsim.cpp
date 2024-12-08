/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include "fsim.h"
#include "../img_params.h"

IQM::GPU::FSIM::FSIM(const VulkanRuntime &runtime):
lowpassFilter(runtime),
logGaborFilter(runtime),
angularFilter(runtime),
combinations(runtime),
sumFilterResponses(runtime),
noise_power(runtime),
estimateEnergy(runtime),
phaseCongruency(runtime),
final_multiply(runtime)
{
    this->downscaleKernel = runtime.createShaderModule("../shaders_out/fsim_downsample.spv");
    this->kernelGradientMap = runtime.createShaderModule("../shaders_out/fsim_gradientmap.spv");
    this->kernelExtractLuma = runtime.createShaderModule("../shaders_out/fsim_extractluma.spv");

    const std::vector layout_2 = {
        *runtime._descLayoutTwoImage,
    };

    const std::vector layout_imbuf = {
        *runtime._descLayoutImageBuffer,
    };

    const std::vector allLayouts = {
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutTwoImage,
        *runtime._descLayoutImageBuffer,
        *runtime._descLayoutImageBuffer,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = runtime._descPool,
        .descriptorSetCount = static_cast<uint32_t>(allLayouts.size()),
        .pSetLayouts = allLayouts.data()
    };

    auto sets = vk::raii::DescriptorSets{runtime._device, descriptorSetAllocateInfo};
    this->descSetDownscaleIn = std::move(sets[0]);
    this->descSetDownscaleRef = std::move(sets[1]);
    this->descSetGradientMapIn = std::move(sets[2]);
    this->descSetGradientMapRef = std::move(sets[3]);
    this->descSetExtractLumaIn = std::move(sets[4]);
    this->descSetExtractLumaRef = std::move(sets[5]);

    // 1x int - kernel size
    const auto downsampleRanges = VulkanRuntime::createPushConstantRange(sizeof(int));

    this->layoutDownscale = runtime.createPipelineLayout(layout_2, downsampleRanges);
    this->pipelineDownscale = runtime.createComputePipeline(this->downscaleKernel, this->layoutDownscale);

    this->layoutGradientMap = runtime.createPipelineLayout(layout_2, {});
    this->pipelineGradientMap = runtime.createComputePipeline(this->kernelGradientMap, this->layoutGradientMap);

    this->layoutExtractLuma = runtime.createPipelineLayout(layout_imbuf, {});
    this->pipelineExtractLuma = runtime.createComputePipeline(this->kernelExtractLuma, this->layoutExtractLuma);
}

IQM::GPU::FSIMResult IQM::GPU::FSIM::computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    FSIMResult result;

    const int F = computeDownscaleFactor(image.width, image.height);

    result.timestamps.mark("downscale factor computed");

    this->sendImagesToGpu(runtime, image, ref);

    result.timestamps.mark("images sent to gpu");

    const auto widthDownscale = static_cast<int>(std::round(static_cast<float>(image.width) / static_cast<float>(F)));
    const auto heightDownscale = static_cast<int>(std::round(static_cast<float>(image.height) / static_cast<float>(F)));

    this->initFftLibrary(runtime, widthDownscale, heightDownscale);
    result.timestamps.mark("FFT library initialized");

    // parallel execution of these steps is possible, so use it
    {
        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        runtime._cmd_buffer->begin(beginInfo);

        this->createDownscaledImages(runtime, widthDownscale, heightDownscale);
        this->computeDownscaledImages(runtime, F, widthDownscale, heightDownscale);
        this->lowpassFilter.constructFilter(runtime, widthDownscale, heightDownscale);
    }

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        nullptr,
        nullptr
    );

    // parallel execution of these steps is possible, so use it
    {
        this->createGradientMap(runtime, widthDownscale, heightDownscale);
        this->logGaborFilter.constructFilter(runtime, this->lowpassFilter.imageLowpassFilter, widthDownscale, heightDownscale);
        this->angularFilter.constructFilter(runtime, widthDownscale, heightDownscale);

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

        result.timestamps.mark("images downscaled + lowpass filter, gradients, log gabor and angular filters computed");
    }

    this->computeFft(runtime, widthDownscale, heightDownscale);
    result.timestamps.mark("fft computed");

    this->combinations.combineFilters(runtime, this->angularFilter, this->logGaborFilter, this->bufferFft, widthDownscale, heightDownscale);
    result.timestamps.mark("filters combined");

    this->computeMassInverseFft(runtime, this->combinations.fftBuffer);
    result.timestamps.mark("mass ifft computed");

    this->sumFilterResponses.computeSums(runtime, this->combinations.fftBuffer, widthDownscale, heightDownscale);
    result.timestamps.mark("filter responses computed");

    this->noise_power.computeNoisePower(runtime, this->combinations.noiseLevels, this->combinations.fftBuffer, widthDownscale, heightDownscale);
    result.timestamps.mark("noise powers computed");

    this->estimateEnergy.estimateEnergy(runtime, this->combinations.fftBuffer, widthDownscale, heightDownscale);
    result.timestamps.mark("noise energy computed");

    this->phaseCongruency.compute(runtime, this->noise_power.noisePowers, this->estimateEnergy.energyBuffers, this->sumFilterResponses.filterResponsesInput, this->sumFilterResponses.filterResponsesRef, widthDownscale, heightDownscale);
    result.timestamps.mark("phase congruency computed");

    auto metrics = this->final_multiply.computeMetrics(
        runtime,
        {this->imageInputDownscaled, this->imageRefDownscaled},
        {this->imageGradientMapInput, this->imageGradientMapRef},
        {this->phaseCongruency.pcInput, this->phaseCongruency.pcRef},
        widthDownscale,
        heightDownscale
    );
    result.timestamps.mark("FSIM, FSIMc computed");

    result.fsim = metrics.first;
    result.fsimc = metrics.second;

    this->teardownFftLibrary();

    return result;
}

int IQM::GPU::FSIM::computeDownscaleFactor(const int width, const int height) {
    auto smallerDim = std::min(width, height);
    return std::max(1, static_cast<int>(std::round(smallerDim / 256.0)));
}

void IQM::GPU::FSIM::sendImagesToGpu(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref) {
    const auto size = image.width * image.height * 4;
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

    auto imageParameters = ImageParameters(image.width, image.height);

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, image.data.data(), imageParameters.height * imageParameters.width * 4);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, imageParameters.height * imageParameters.width * 4, {});
    memcpy(inBufData, ref.data.data(), imageParameters.height * imageParameters.width * 4);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(image.width, image.height, 1),
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

    this->imageInput = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));
    this->imageRef = std::make_shared<VulkanImage>(runtime.createImage(srcImageInfo));

    // copy data to images, correct formats
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_bufferTransfer->begin(beginInfo);

    runtime.setImageLayout(runtime._cmd_bufferTransfer, this->imageInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_bufferTransfer, this->imageRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = imageParameters.width,
        .bufferImageHeight = imageParameters.height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{imageParameters.width, imageParameters.height, 1}
    };
    runtime._cmd_bufferTransfer->copyBufferToImage(stgBuf, this->imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    runtime._cmd_bufferTransfer->copyBufferToImage(stgRefBuf, this->imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    runtime._cmd_bufferTransfer->end();

    const std::vector cmdBufsCopy = {
        &**runtime._cmd_bufferTransfer
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{runtime._device, vk::FenceCreateInfo{}};

    runtime._transferQueue->submit(submitInfoCopy, *fenceCopy);
    runtime.waitForFence(fenceCopy);
}

void IQM::GPU::FSIM::createDownscaledImages(const VulkanRuntime &runtime, int width_downscale, int height_downscale) {
    const vk::ImageCreateInfo imageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = vk::Extent3D(width_downscale, height_downscale, 1),
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

    vk::ImageCreateInfo imageFloatInfo = {imageInfo};
    imageFloatInfo.format = vk::Format::eR32Sfloat;
    imageFloatInfo.usage = vk::ImageUsageFlagBits::eStorage;

    vk::ImageCreateInfo imageFftInfo = {imageFloatInfo};
    imageFftInfo.format = vk::Format::eR32G32Sfloat;
    imageFftInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;

    this->imageInputDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->imageRefDownscaled = std::make_shared<VulkanImage>(runtime.createImage(imageInfo));
    this->imageGradientMapInput = std::make_shared<VulkanImage>(runtime.createImage(imageFloatInfo));
    this->imageGradientMapRef = std::make_shared<VulkanImage>(runtime.createImage(imageFloatInfo));

    auto imageInfosInput = VulkanRuntime::createImageInfos({
        this->imageInput,
        this->imageInputDownscaled,
    });

    auto imageInfosRef = VulkanRuntime::createImageInfos({
        this->imageRef,
        this->imageRefDownscaled,
    });

    auto imageInfosGradIn = VulkanRuntime::createImageInfos({
        this->imageInputDownscaled,
        this->imageGradientMapInput,
    });

    auto imageInfosGradRef = VulkanRuntime::createImageInfos({
        this->imageRefDownscaled,
        this->imageGradientMapRef,
    });

    const vk::WriteDescriptorSet writeSetInput{
        .dstSet = this->descSetDownscaleIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosInput.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetRef{
        .dstSet = this->descSetDownscaleRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetGradIn{
        .dstSet = this->descSetGradientMapIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosGradIn.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetGradRef{
        .dstSet = this->descSetGradientMapRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 2,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imageInfosGradRef.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetRef, writeSetInput, writeSetGradIn, writeSetGradRef
    };

    runtime._device.updateDescriptorSets(writes, nullptr);
}

void IQM::GPU::FSIM::computeDownscaledImages(const VulkanRuntime &runtime, const int F, const int width, const int height) {
    runtime.setImageLayout(runtime._cmd_buffer, this->imageInputDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->imageRefDownscaled->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineDownscale);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleIn}, {});

    runtime._cmd_buffer->pushConstants<int>(this->layoutDownscale, vk::ShaderStageFlagBits::eCompute, 0, F);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutDownscale, 0, {this->descSetDownscaleRef}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FSIM::createGradientMap(const VulkanRuntime &runtime, int width, int height) {
    runtime.setImageLayout(runtime._cmd_buffer, this->imageGradientMapInput->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    runtime.setImageLayout(runtime._cmd_buffer, this->imageGradientMapRef->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineGradientMap);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapIn}, {});

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutGradientMap, 0, {this->descSetGradientMapRef}, {});

    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);
}

void IQM::GPU::FSIM::initFftLibrary(const VulkanRuntime &runtime, const int width, const int height) {
    // image size * 2 float components (complex numbers) * 2 batches
    uint64_t bufferSize = width * height * sizeof(float) * 2 * 2;

    VkFFTApplication fftApp = {};

    VkFFTConfiguration fftConfig = {};
    fftConfig.FFTdim = 2;
    fftConfig.size[0] = width;
    fftConfig.size[1] = height;
    fftConfig.bufferSize = &bufferSize;

    VkDevice deviceRef = *runtime._device;
    VkPhysicalDevice physDeviceRef = *runtime._physicalDevice;
    VkQueue queueRef = **runtime._queue;
    VkCommandPool cmdPoolRef = **runtime._commandPool;
    fftConfig.physicalDevice = &physDeviceRef;
    fftConfig.device = &deviceRef;
    fftConfig.queue = &queueRef;
    fftConfig.commandPool = &cmdPoolRef;
    fftConfig.numberBatches = 2;
    fftConfig.makeForwardPlanOnly = true;

    this->fftFence =  vk::raii::Fence{runtime._device, vk::FenceCreateInfo{}};
    VkFence fenceRef = *this->fftFence;
    fftConfig.fence = &fenceRef;

    if (initializeVkFFT(&fftApp, fftConfig) != VKFFT_SUCCESS) {
        throw std::runtime_error("failed to initialize FFT");
    }

    this->fftApplication = fftApp;

    // (image size * 2 float components (complex numbers) ) * 16 filters * 3 cases (by itself, times input, times reference)
    uint64_t bufferSizeInverse = width * height * sizeof(float) * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3;

    VkFFTApplication fftAppInverse = {};

    VkFFTConfiguration fftConfigInverse = {};
    fftConfigInverse.FFTdim = 2;
    fftConfigInverse.size[0] = width;
    fftConfigInverse.size[1] = height;
    fftConfigInverse.bufferSize = &bufferSizeInverse;

    fftConfigInverse.physicalDevice = &physDeviceRef;
    fftConfigInverse.device = &deviceRef;
    fftConfigInverse.queue = &queueRef;
    fftConfigInverse.commandPool = &cmdPoolRef;
    fftConfigInverse.numberBatches = 16 * 3;
    fftConfigInverse.makeInversePlanOnly = true;
    fftConfigInverse.normalize = true;
    fftConfigInverse.isCompilerInitialized = true;

    this->fftFenceInverse =  vk::raii::Fence{runtime._device, vk::FenceCreateInfo{}};
    VkFence fenceRefInverse = *this->fftFenceInverse;
    fftConfigInverse.fence = &fenceRefInverse;

    if (initializeVkFFT(&fftAppInverse, fftConfigInverse) != VKFFT_SUCCESS) {
        throw std::runtime_error("failed to initialize FFT");
    }

    this->fftApplicationInverse = fftAppInverse;
}

void IQM::GPU::FSIM::teardownFftLibrary() {
    deleteVkFFT(&this->fftApplicationInverse);
    deleteVkFFT(&this->fftApplication);
}

void IQM::GPU::FSIM::computeFft(const VulkanRuntime &runtime, const int width, const int height) {
    // image size * 2 float components (complex numbers) * 2 batches
    uint64_t bufferSize = width * height * sizeof(float) * 2 * 2;

    auto [fftBuf, fftMem] = runtime.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    fftBuf.bindMemory(fftMem, 0);

    this->memoryFft = std::move(fftMem);
    this->bufferFft = std::move(fftBuf);

    std::vector bufIn = {
        vk::DescriptorBufferInfo{
            .buffer = this->bufferFft,
            .offset = 0,
            .range = bufferSize / 2,
        }
    };
    auto imInfoInImage = VulkanRuntime::createImageInfos({
        this->imageInputDownscaled,
    });
    const vk::WriteDescriptorSet writeSetInImage{
        .dstSet = this->descSetExtractLumaIn,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imInfoInImage.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetInBuf{
        .dstSet = this->descSetExtractLumaIn,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufIn.data(),
        .pTexelBufferView = nullptr,
    };

    std::vector bufRef = {
        vk::DescriptorBufferInfo{
            .buffer = this->bufferFft,
            .offset = bufferSize / 2,
            .range = bufferSize / 2,
        }
    };

    auto imInfoRefImage = VulkanRuntime::createImageInfos({
        this->imageRefDownscaled,
    });
    const vk::WriteDescriptorSet writeSetRefImage{
        .dstSet = this->descSetExtractLumaRef,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .pImageInfo = imInfoRefImage.data(),
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    const vk::WriteDescriptorSet writeSetRefBuf{
        .dstSet = this->descSetExtractLumaRef,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pImageInfo = nullptr,
        .pBufferInfo = bufRef.data(),
        .pTexelBufferView = nullptr,
    };

    const std::vector writes = {
        writeSetInImage, writeSetInBuf, writeSetRefImage, writeSetRefBuf
    };

    runtime._device.updateDescriptorSets(writes, nullptr);

    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    //shader works in 8x8 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(width, height, 8);

    runtime._cmd_buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineExtractLuma);
    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaIn}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    runtime._cmd_buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutExtractLuma, 0, {this->descSetExtractLumaRef}, {});
    runtime._cmd_buffer->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    runtime._cmd_buffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {barrier},
        nullptr,
        nullptr
    );

    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **runtime._cmd_buffer;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = *this->bufferFft;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(&this->fftApplication, -1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append FFT: " + std::to_string(res);
        throw std::runtime_error(err);
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
}

void IQM::GPU::FSIM::computeMassInverseFft(const VulkanRuntime &runtime, const vk::raii::Buffer &buffer) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    runtime._cmd_buffer->begin(beginInfo);

    VkFFTLaunchParams launchParams = {};
    VkCommandBuffer cmdBuf = **runtime._cmd_buffer;
    launchParams.commandBuffer = &cmdBuf;
    VkBuffer fftBufRef = *buffer;
    launchParams.buffer = &fftBufRef;

    if (auto res = VkFFTAppend(&this->fftApplicationInverse, 1, &launchParams); res != VKFFT_SUCCESS) {
        std::string err = "failed to append inverse FFT: " + std::to_string(res);
        throw std::runtime_error(err);
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
}

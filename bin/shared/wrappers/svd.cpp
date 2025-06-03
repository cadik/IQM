/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "svd.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"

void IQM::Bin::svd_run(const IQM::Bin::Args &args, const IQM::VulkanInstance &instance, const std::vector<Match> &imageMatches) {
    IQM::SVD svd(*instance.device());

    int processed = 0;

    for (const auto& match : imageMatches) {
        try {
            VulkanResource::resetMemCounter();
            Timestamps timestamps;
            auto start = std::chrono::high_resolution_clock::now();

            const auto input = load_image(match.testPath);
            const auto reference = load_image(match.refPath);
            if (input.height != reference.height || input.width != reference.width) {
                throw std::runtime_error("Test and reference images have different sizes");
            }

            timestamps.mark("images loaded");

            initRenderDoc();

            auto res = svd_init_res(input, reference, instance);
            timestamps.mark("resources allocated");

            svd_upload(instance, res);

            auto svdArgs = IQM::SVDInput {
                .device = instance.device(),
                .cmdBuf = &*instance.cmdBuf(),
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .ivConvTest = &res.imagesFloat[0]->imageView,
                .ivConvRef = &res.imagesFloat[1]->imageView,
                .bufSvd = &res.svdBuf,
                .bufReduce = &res.reduceBuf,
                .bufSort = &res.sortBuf,
                .bufSortTemp = &res.sortTempBuf,
                .width = input.width,
                .height = input.height
            };

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmdBuf()->begin(beginInfo);

            svd.computeMetric(svdArgs);

            instance.cmdBuf()->end();

            const std::vector cmdBufs = {
                &**instance.cmdBuf()
            };

            auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
            const vk::SubmitInfo submitInfo{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*res.uploadDone,
                .pWaitDstStageMask = &mask,
                .commandBufferCount = 1,
                .pCommandBuffers = *cmdBufs.data(),
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*res.computeDone
            };

            instance.queue()->submit(submitInfo, {});
            timestamps.mark("submit compute GPU pipeline");
            // wait so cmd buffer can be reused for GPU -> CPU transfer
            instance.waitForFence(res.transferFence);

            auto result = svd_copy_back(instance, res, timestamps, input.width/8 * input.height/8);

            finishRenderDoc();

            if (match.outPath.has_value()) {
                save_float_image(args.outputPath.value(), result.imageData, input.width, input.height);
            }

            timestamps.mark("output saved");

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.msvd << std::endl;
            if (args.verbose) {
                timestamps.print(start, end);
                double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
                std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to process '" << match.testPath << "': " << e.what() << std::endl;
            continue;
        }

        processed += 1;
    }

    std::cout << "Processed " << processed << "/" << imageMatches.size() <<" images" << std::endl;
}

void IQM::Bin::svd_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::SVD &svd, const IQM::Bin::InputImage &input, const IQM::Bin::InputImage &ref) {
    try {
        VulkanResource::resetMemCounter();
        Timestamps timestamps;
        auto start = std::chrono::high_resolution_clock::now();

        timestamps.mark("images loaded");

        initRenderDoc();

        auto res = svd_init_res(input, ref, instance);
        timestamps.mark("resources allocated");

        svd_upload(instance, res);

        auto svdArgs = IQM::SVDInput {
            .device = instance.device(),
            .cmdBuf = &*instance.cmdBuf(),
            .ivTest = &res.imageInput->imageView,
            .ivRef = &res.imageRef->imageView,
            .ivConvTest = &res.imagesFloat[0]->imageView,
            .ivConvRef = &res.imagesFloat[1]->imageView,
            .bufSvd = &res.svdBuf,
            .bufReduce = &res.reduceBuf,
            .bufSort = &res.sortBuf,
            .bufSortTemp = &res.sortTempBuf,
            .width = input.width,
            .height = input.height
        };

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        instance.cmdBuf()->begin(beginInfo);

        svd.computeMetric(svdArgs);

        instance.cmdBuf()->end();

        const std::vector cmdBufs = {
            &**instance.cmdBuf()
        };

        auto mask = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eComputeShader};
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*res.uploadDone,
            .pWaitDstStageMask = &mask,
            .commandBufferCount = 1,
            .pCommandBuffers = *cmdBufs.data(),
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*res.computeDone
        };

        instance.queue()->submit(submitInfo, {});
        timestamps.mark("submit compute GPU pipeline");
        // wait so cmd buffer can be reused for GPU -> CPU transfer
        instance.waitForFence(res.transferFence);

        auto result = svd_copy_back(instance, res, timestamps, input.width/8 * input.height/8);

        finishRenderDoc();

        timestamps.mark("output saved");

        const auto end = std::chrono::high_resolution_clock::now();

        if (args.verbose) {
            std::cout << args.inputPath << ": " << result.msvd << std::endl;
            timestamps.print(start, end);

            double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
            std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to process '" << args.inputPath << "': " << e.what() << std::endl;
    }
}

IQM::Bin::SVDResources IQM::Bin::svd_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance &instance) {
    // always 4 channels on input, with 1B per channel
    const auto size = (test.width * test.height) * 4;
    const auto downSize = (test.width/8 * test.height/8) * 4;
    const auto downSizeSvd = (test.width/8 * test.height/8) * 4 * 8 * 2;
    auto [stgBuf, stgMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        size,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached
    );
    auto [stgRefBuf, stgRefMem] = VulkanResource::createBuffer(
    *instance.device(),
    *instance.physicalDevice(),
        size,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [svdBuf, svdMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        downSizeSvd,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    auto [reduceBuf, reduceMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        downSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    auto [sortBuf, sortMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        downSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    auto [sortTempBuf, sortTempMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        downSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    svdBuf.bindMemory(svdMem, 0);
    reduceBuf.bindMemory(reduceMem, 0);
    sortBuf.bindMemory(sortMem, 0);
    sortTempBuf.bindMemory(sortTempMem, 0);

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, test.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(test.width, test.height, 1),
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

    vk::ImageCreateInfo intermediateImageInfo = {srcImageInfo};
    intermediateImageInfo.usage = vk::ImageUsageFlagBits::eStorage;
    intermediateImageInfo.format = vk::Format::eR32Sfloat;

    vk::ImageCreateInfo dstImageInfo = {srcImageInfo};
    dstImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    dstImageInfo.format = vk::Format::eR32Sfloat;

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto imagesFloat = std::vector<std::shared_ptr<VulkanImage>>();

    for (int i = 0; i < 2; i++) {
        imagesFloat.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), intermediateImageInfo)));
    }

    return SVDResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .svdBuf = std::move(svdBuf),
        .svdMemory = std::move(svdMem),
        .reduceBuf = std::move(reduceBuf),
        .reduceMemory = std::move(reduceMem),
        .sortBuf = std::move(sortBuf),
        .sortMemory = std::move(sortMem),
        .sortTempBuf = std::move(sortTempBuf),
        .sortTempMemory = std::move(sortTempMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imagesFloat = imagesFloat,
        .uploadDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device()->createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::svd_upload(const IQM::VulkanInstance &instance, const SVDResources &res) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
    };

    imagesToInit.insert(imagesToInit.end(), res.imagesFloat.begin(), res.imagesFloat.end());

    VulkanResource::initImages(*instance.cmdBufTransfer(), imagesToInit);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = res.imageInput->width,
        .bufferImageHeight = res.imageInput->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
    };
    instance.cmdBufTransfer()->copyBufferToImage(res.stgInput, res.imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmdBufTransfer()->copyBufferToImage(res.stgRef, res.imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);

    instance.cmdBufTransfer()->end();

    const std::vector cmdBufsCopy = {
        &**instance.cmdBufTransfer()
    };

    const vk::SubmitInfo submitInfoCopy{
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*res.uploadDone
    };

    instance.queueTransfer()->submit(submitInfoCopy, res.transferFence);
}

IQM::Bin::SVDResult IQM::Bin::svd_copy_back(const IQM::VulkanInstance &instance, const SVDResources &res, Timestamps &timestamps, uint32_t pixelCount) {
    SVDResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfoCopy);

    vk::BufferCopy copyRegion{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeof(float) * pixelCount,
    };
    instance.cmdBufTransfer()->copyBuffer(res.reduceBuf, res.stgInput, copyRegion);
    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = sizeof(float) * pixelCount,
        .size = sizeof(float),
    };
    instance.cmdBufTransfer()->copyBuffer(res.sortTempBuf, res.stgInput, bufCopy);

    instance.cmdBufTransfer()->end();

    const std::vector cmdBufsCopy = {
        &**instance.cmdBufTransfer()
    };

    auto maskCopy = vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTransfer};
    const vk::SubmitInfo submitInfoCopy{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*res.computeDone,
        .pWaitDstStageMask = &maskCopy,
        .commandBufferCount = 1,
        .pCommandBuffers = *cmdBufsCopy.data()
    };

    const vk::raii::Fence fenceCopy{*instance.device(), vk::FenceCreateInfo{}};

    instance.queueTransfer()->submit(submitInfoCopy, *fenceCopy);
    instance.device()->waitIdle();

    timestamps.mark("end GPU work");

    std::vector<float> outputData(pixelCount);
    void * outBufData = res.stgInputMemory.mapMemory(0, (pixelCount + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, pixelCount * sizeof(float));
    result.msvd = (static_cast<float*>(outBufData))[pixelCount];
    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    result.imageData = std::move(outputData);

    return result;
}

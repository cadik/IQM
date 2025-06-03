/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "lpips.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"
#include "IQM/base/colorize.h"
#include "IQM/base/viridis.h"

void IQM::Bin::lpips_run(const IQM::Bin::Args &args, const IQM::VulkanInstance &instance, const std::vector<Match> &imageMatches) {
    IQM::LPIPS lpips(*instance.device());
    IQM::Colorize colorizer(*instance.device());

    VulkanResource::resetMemCounter();

    auto modelData = load_model("lpips.dat");
    auto model = lpips_load_model(instance, lpips.modelSize(), modelData);
    auto modelSize = VulkanResource::memCounter();

    int processed = 0;

    for (const auto& match : imageMatches) {
        try {
            VulkanResource::resetMemCounter();
            // model is shared
            VulkanResource::addMemCounter(modelSize);
            Timestamps timestamps;
            auto start = std::chrono::high_resolution_clock::now();

            const auto input = load_image(match.testPath);
            const auto reference = load_image(match.refPath);
            if (input.height != reference.height || input.width != reference.width) {
                throw std::runtime_error("Test and reference images have different sizes");
            }

            timestamps.mark("images loaded");

            initRenderDoc();

            auto sizes = lpips.bufferSizes(input.width, input.height);

            auto res = lpips_init_res(input, reference, instance, sizes, args.outputPath.has_value(), args.colorize);
            timestamps.mark("resources allocated");

            lpips_upload(instance, res, model, lpips.modelSize(), args.outputPath.has_value(), args.colorize);

            auto lpipsArgs = IQM::LPIPSInput {
                .device = instance.device(),
                .cmdBuf = &*instance.cmdBuf(),
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .imgOut = nullptr,
                .bufWeights = &model.weightsBuf,
                .bufTest = &res.convInputBuf,
                .bufRef = &res.convRefBuf,
                .bufComp = &res.compareBuf,
                .width = input.width,
                .height = input.height,
            };

            if (res.imageOut != nullptr) {
                lpipsArgs.imgOut = &res.imageOut->image;
            }

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmdBuf()->begin(beginInfo);

            lpips.computeMetric(lpipsArgs);

            if (args.outputPath.has_value()) {
                if (args.colorize) {
                    auto colorizerInput = IQM::ColorizeInput{
                        .device = instance.device(),
                        .cmdBuf = &*instance.cmdBuf(),
                        .ivIn = &res.imageOut->imageView,
                        .ivOut = &res.imageInput->imageView,
                        .ivColormap = &res.imageColorMap->imageView,
                        .invert = false,
                        .scaler = 4.0,
                        .width = input.width,
                        .height = input.height
                    };

                    colorizer.compute(colorizerInput);
                } else {
                    std::array offsets = {
                        vk::Offset3D{0, 0, 0},
                        vk::Offset3D{static_cast<int>(res.imageInput->width), static_cast<int>(res.imageInput->height), 1}
                    };
                    // copy RGBA f32 to RGBA u8
                    std::vector region {
                        vk::ImageBlit {
                            .srcSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                            .srcOffsets = offsets,
                            .dstSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                            .dstOffsets = offsets,
                        }
                    };
                    instance.cmdBuf()->blitImage(res.imageOut->image, vk::ImageLayout::eGeneral, res.imageExport->image, vk::ImageLayout::eGeneral, {region}, vk::Filter::eNearest);
                }
            }

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

            auto result = lpips_copy_back(instance, res, timestamps, args.outputPath.has_value(), args.colorize);

            finishRenderDoc();
            if (match.outPath.has_value()) {
                if (args.colorize) {
                    save_color_image(args.outputPath.value(), result.imageData, input.width, input.height);
                } else {
                    save_char_image(args.outputPath.value(), result.imageData, input.width, input.height);
                }
            }

            timestamps.mark("output saved");

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.distance << std::endl;
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

void IQM::Bin::lpips_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::LPIPS &lpips, const IQM::Bin::InputImage &input, const IQM::Bin::InputImage &ref, const std::vector<float> &lpipsModel) {
    try {
        VulkanResource::resetMemCounter();
        IQM::Colorize colorizer(*instance.device());

        Timestamps timestamps;
        auto start = std::chrono::high_resolution_clock::now();

        timestamps.mark("images loaded");

        initRenderDoc();

        auto sizes = lpips.bufferSizes(input.width, input.height);

        auto res = lpips_init_res(input, ref, instance, sizes, true, args.colorize);
        auto model = lpips_load_model(instance, lpips.modelSize(), lpipsModel);
        timestamps.mark("resources allocated");

        lpips_upload(instance, res, model, lpips.modelSize(), true, args.colorize);

        auto lpipsArgs = IQM::LPIPSInput {
            .device = instance.device(),
            .cmdBuf = &*instance.cmdBuf(),
            .ivTest = &res.imageInput->imageView,
            .ivRef = &res.imageRef->imageView,
            .imgOut = &res.imageOut->image,
            .bufWeights = &model.weightsBuf,
            .bufTest = &res.convInputBuf,
            .bufRef = &res.convRefBuf,
            .bufComp = &res.compareBuf,
            .width = input.width,
            .height = input.height,
        };

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        instance.cmdBuf()->begin(beginInfo);

        lpips.computeMetric(lpipsArgs);

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

        auto result = lpips_copy_back(instance, res, timestamps, false, args.colorize);

        finishRenderDoc();

        timestamps.mark("output saved");

        const auto end = std::chrono::high_resolution_clock::now();

        if (args.verbose) {
            std::cout << args.inputPath << ": " << result.distance << std::endl;
            timestamps.print(start, end);

            double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
            std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to process '" << args.inputPath << "': " << e.what() << std::endl;
    }
}

IQM::Bin::LPIPSResources IQM::Bin::lpips_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance &instance, const LPIPSBufferSizes &bufferSizes, const bool hasOutput, const bool colorize) {
    // always 4 channels on input, with 1B per channel
    const auto size = (test.width * test.height + 1) * 4;
    const auto colormapSize = 256 * 4 * sizeof(float);
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
    auto [cmBuf, cmMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        colormapSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    auto [convInputBuf, convInputMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        bufferSizes.bufTest,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    auto [convRefBuf, convRefMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        bufferSizes.bufRef,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    auto [compBuf, compMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        bufferSizes.bufComp,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    cmBuf.bindMemory(cmMem, 0);
    convInputBuf.bindMemory(convInputMem, 0);
    convRefBuf.bindMemory(convRefMem, 0);
    compBuf.bindMemory(compMem, 0);

    void * inBufData = stgMem.mapMemory(0, size, {});
    memcpy(inBufData, test.data.data(), size);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, size, {});
    memcpy(inBufData, ref.data.data(), size);
    stgRefMem.unmapMemory();

    inBufData = cmMem.mapMemory(0, colormapSize, {});
    memcpy(inBufData, viridis, colormapSize);
    cmMem.unmapMemory();

    vk::ImageCreateInfo srcImageInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D(test.width, test.height, 1),
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    vk::ImageCreateInfo colorMapImageInfo = {srcImageInfo};
    colorMapImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
    colorMapImageInfo.extent = vk::Extent3D(256, 1, 1);
    colorMapImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo exitImageInfo = {srcImageInfo};
    exitImageInfo.format = vk::Format::eR8Unorm;

    vk::ImageCreateInfo dstImageInfo = {srcImageInfo};
    dstImageInfo.format = vk::Format::eR32Sfloat;

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));

    auto imageOut = std::shared_ptr<VulkanImage>();
    auto imageExport = std::shared_ptr<VulkanImage>();
    auto imageColorMap = std::shared_ptr<VulkanImage>();
    if (hasOutput) {
        imageOut = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), dstImageInfo));
        if (colorize) {
            imageColorMap = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), colorMapImageInfo));
        } else {
            imageExport = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), exitImageInfo));
        }
    }

    return LPIPSResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .stgColormap = std::move(cmBuf),
        .stgColormapMemory = std::move(cmMem),
        .convInputBuf = std::move(convInputBuf),
        .convInputMemory = std::move(convInputMem),
        .convRefBuf = std::move(convRefBuf),
        .convRefMemory = std::move(convRefMem),
        .compareBuf = std::move(compBuf),
        .compareMemory = std::move(compMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imageOut = imageOut,
        .imageExport = imageExport,
        .imageColorMap = imageColorMap,
        .uploadDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device()->createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::lpips_upload(const IQM::VulkanInstance &instance, const LPIPSResources &res, const LPIPSModelResources &model, const unsigned long modelSize, const bool hasOutput, const bool colorize) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
    };

    if (hasOutput) {
        imagesToInit.push_back(res.imageOut);
        if (colorize) {
            imagesToInit.push_back(res.imageColorMap);
        } else {
            imagesToInit.push_back(res.imageExport);
        }
    }

    VulkanResource::initImages(*instance.cmdBufTransfer(), imagesToInit);

    vk::BufferCopy modelRegion {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = modelSize,
    };
    instance.cmdBufTransfer()->copyBuffer(model.stgWeights, model.weightsBuf, {modelRegion});

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
    if (hasOutput && colorize) {
        vk::BufferImageCopy copyColorMapRegion{
            .bufferOffset = 0,
            .bufferRowLength = 256,
            .bufferImageHeight = 1,
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{256, 1, 1}
        };
        instance.cmdBufTransfer()->copyBufferToImage(res.stgColormap, res.imageColorMap->image,  vk::ImageLayout::eGeneral, copyColorMapRegion);
    }

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

IQM::Bin::LPIPSModelResources IQM::Bin::lpips_load_model(const IQM::VulkanInstance &instance, const unsigned long modelSize, const std::vector<float> &modelData) {
    auto [stgWeightBuf, stgWeightMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        modelSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    auto [weightBuf, weightMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        modelSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgWeightBuf.bindMemory(stgWeightMem, 0);
    weightBuf.bindMemory(weightMem, 0);

    void * inBufData = stgWeightMem.mapMemory(0, modelSize, {});
    memcpy(inBufData, modelData.data(), modelSize);
    stgWeightMem.unmapMemory();

    return LPIPSModelResources{
        .stgWeights = std::move(stgWeightBuf),
        .stgWeightsMemory = std::move(stgWeightMem),
        .weightsBuf = std::move(weightBuf),
        .weightsMemory = std::move(weightMem),
    };
}

IQM::Bin::LPIPSResult IQM::Bin::lpips_copy_back(const VulkanInstance &instance, const LPIPSResources &res, Timestamps &timestamps, bool hasOutput, bool colorize) {
    LPIPSResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfoCopy);

    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeof(float),
    };
    instance.cmdBufTransfer()->copyBuffer(res.convInputBuf, res.stgInput, bufCopy);

    if (hasOutput) {
        vk::BufferImageCopy copyRegion{
            .bufferOffset = sizeof(float),
            .bufferRowLength = res.imageInput->width,
            .bufferImageHeight = res.imageInput->height,
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = vk::Offset3D{0, 0, 0},
            .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
        };

        if (colorize) {
            instance.cmdBufTransfer()->copyImageToBuffer(res.imageInput->image, vk::ImageLayout::eGeneral, res.stgInput, {copyRegion});
        } else {
            instance.cmdBufTransfer()->copyImageToBuffer(res.imageExport->image, vk::ImageLayout::eGeneral, res.stgInput, {copyRegion});
        }
    }

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

    void * outBufData = res.stgInputMemory.mapMemory(0, sizeof(float) + res.imageInput->width * res.imageInput->height * 4, {});
    memcpy(&result.distance, outBufData, sizeof(float));

    if (hasOutput) {
        if (colorize) {
            std::vector<unsigned char> data(res.imageInput->width * res.imageInput->height * 4);
            memcpy(data.data(), outBufData + sizeof(float), res.imageInput->height * res.imageInput->width * 4 * sizeof(unsigned char));
            result.imageData = std::move(data);
        } else {
            std::vector<unsigned char> data(res.imageInput->width * res.imageInput->height);
            memcpy(data.data(), outBufData + sizeof(float), res.imageInput->height * res.imageInput->width * sizeof(unsigned char));
            result.imageData = std::move(data);
        }
    }

    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    return result;
}

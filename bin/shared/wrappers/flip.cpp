/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "flip.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"
#include "IQM/base/viridis.h"
#include "IQM/base/colorize.h"

using IQM::VulkanInstance;

void IQM::Bin::flip_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches) {
    IQM::FLIP flip(*instance.device());
    IQM::Colorize colorizer(*instance.device());

    FLIPArguments flipArgs;
    if (args.options.contains("--flip-width")) {
        flipArgs.monitor_width = std::stof(args.options.at("--flip-width"));
    }
    if (args.options.contains("--flip-res")) {
        flipArgs.monitor_resolution_x = std::stof(args.options.at("--flip-res"));
    }
    if (args.options.contains("--flip-distance")) {
        flipArgs.monitor_distance = std::stof(args.options.at("--flip-distance"));
    }

    if (args.verbose) {
        std::cout << "FLIP monitor resolution: "<< flipArgs.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flipArgs.monitor_distance << std::endl
        << "FLIP monitor width: "<< flipArgs.monitor_width << std::endl;
    }

    unsigned featureKernelSize = IQM::FLIP::featureKernelSize(flipArgs);

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

            auto res = flip_init_res(input, reference, instance, args.colorize, featureKernelSize);
            timestamps.mark("resources allocated");

            flip_upload(instance, res);

            auto flipInput = IQM::FLIPInput {
                .args = flipArgs,
                .device = instance.device(),
                .cmdBuf = &*instance.cmdBuf(),
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .ivOut = &res.imageOut->imageView,
                .ivFeatFilter = &res.imageFeatureFilter->imageView,
                .imgOut = &res.imageOut->image,
                .buffer = &res.buf,
                .width = input.width,
                .height = input.height
            };

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmdBuf()->begin(beginInfo);

            flip.computeMetric(flipInput);

            if (args.colorize) {
                auto colorizerInput = IQM::ColorizeInput{
                    .device = instance.device(),
                    .cmdBuf = &*instance.cmdBuf(),
                    .ivIn = &res.imageOut->imageView,
                    .ivOut = &res.imageInput->imageView,
                    .ivColormap = &res.imageColorMap->imageView,
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
                instance.cmdBuf()->blitImage(res.imageOut->image, vk::ImageLayout::eGeneral, res.imageGreyscaleOut->image, vk::ImageLayout::eGeneral, {region}, vk::Filter::eNearest);
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

            auto result = flip_copy_back(instance, res, timestamps, args.colorize);

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
            std::cout << match.testPath << ": " << result.meanFlip << std::endl;
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

void IQM::Bin::flip_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::FLIP &flip, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref) {
    VulkanResource::resetMemCounter();
    FLIPArguments flipArgs;
    if (args.options.contains("--flip-width")) {
        flipArgs.monitor_width = std::stof(args.options.at("--flip-width"));
    }
    if (args.options.contains("--flip-res")) {
        flipArgs.monitor_resolution_x = std::stof(args.options.at("--flip-res"));
    }
    if (args.options.contains("--flip-distance")) {
        flipArgs.monitor_distance = std::stof(args.options.at("--flip-distance"));
    }

    if (args.verbose) {
        std::cout << "FLIP monitor resolution: "<< flipArgs.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flipArgs.monitor_distance << std::endl
        << "FLIP monitor width: "<< flipArgs.monitor_width << std::endl;
    }

    unsigned featureKernelSize = IQM::FLIP::featureKernelSize(flipArgs);
    IQM::Colorize colorizer(*instance.device());

    try {
        Timestamps timestamps;
        auto start = std::chrono::high_resolution_clock::now();

        timestamps.mark("images loaded");

        initRenderDoc();

        auto res = flip_init_res(input, ref, instance, args.colorize, featureKernelSize);
        timestamps.mark("resources allocated");

        flip_upload(instance, res);

        auto flipInput = IQM::FLIPInput {
            .args = flipArgs,
            .device = instance.device(),
            .cmdBuf = &*instance.cmdBuf(),
            .ivTest = &res.imageInput->imageView,
            .ivRef = &res.imageRef->imageView,
            .ivOut = &res.imageOut->imageView,
            .ivFeatFilter = &res.imageFeatureFilter->imageView,
            .imgOut = &res.imageOut->image,
            .buffer = &res.buf,
            .width = input.width,
            .height = input.height
        };

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        instance.cmdBuf()->begin(beginInfo);

        flip.computeMetric(flipInput);

        if (args.colorize) {
            auto colorizerInput = IQM::ColorizeInput{
                .device = instance.device(),
                .cmdBuf = &*instance.cmdBuf(),
                .ivIn = &res.imageOut->imageView,
                .ivOut = &res.imageInput->imageView,
                .ivColormap = &res.imageColorMap->imageView,
                .width = input.width,
                .height = input.height,
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
            instance.cmdBuf()->blitImage(res.imageOut->image, vk::ImageLayout::eGeneral, res.imageGreyscaleOut->image, vk::ImageLayout::eGeneral, {region}, vk::Filter::eNearest);
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

        auto result = flip_copy_back(instance, res, timestamps, args.colorize);

        finishRenderDoc();

        timestamps.mark("output saved");

        const auto end = std::chrono::high_resolution_clock::now();
        if (args.verbose) {
            std::cout << args.inputPath << ": " << result.meanFlip << std::endl;
            timestamps.print(start, end);

            double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
            std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to process '" << args.inputPath << "': " << e.what() << std::endl;
    }
}

IQM::Bin::FLIPResources IQM::Bin::flip_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance &instance, bool colorize, unsigned featureKernelSize) {
    // always 4 channels on input, with 1B per channel
    // add 1 float to end so buffer can be reused for writeback from GPU
    const auto outSize = ((test.width * test.height) + 1) * sizeof(float);
    const auto size = (test.width * test.height) * sizeof(float);
    const auto sizeIntermediate = (test.width * test.height) * sizeof(float) * 13;
    const auto colormapSize = 256 * 4 * sizeof(float);
    auto [stgBuf, stgMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        outSize,
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
    auto [buf, mem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        sizeIntermediate,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    cmBuf.bindMemory(cmMem, 0);
    buf.bindMemory(mem, 0);

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

    vk::ImageCreateInfo floatImageInfo = {srcImageInfo};
    floatImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    floatImageInfo.format = vk::Format::eR32Sfloat;

    vk::ImageCreateInfo greyscaleImageInfo = {srcImageInfo};
    greyscaleImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    greyscaleImageInfo.format = vk::Format::eR8Unorm;

    vk::ImageCreateInfo featureImageInfo = {srcImageInfo};
    featureImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    featureImageInfo.extent = vk::Extent3D(featureKernelSize, 1, 1);
    featureImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    vk::ImageCreateInfo colorMapImageInfo = {srcImageInfo};
    colorMapImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
    colorMapImageInfo.extent = vk::Extent3D(256, 1, 1);
    colorMapImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto const imageOut = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), floatImageInfo));
    auto const imageFeatureFilter = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), featureImageInfo));
    auto const imageColorMap = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), colorMapImageInfo));

    auto imageGreyscale = std::shared_ptr<VulkanImage>();
    if (!colorize) {
        imageGreyscale = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), greyscaleImageInfo));
    }

    return FLIPResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .stgColormap = std::move(cmBuf),
        .stgColormapMemory = std::move(cmMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imageGreyscaleOut = imageGreyscale,
        .buf = std::move(buf),
        .memory = std::move(mem),
        .imageFeatureFilter = imageFeatureFilter,
        .imageColorMap = imageColorMap,
        .imageOut = imageOut,
        .uploadDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device()->createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::flip_upload(const VulkanInstance &instance, const FLIPResources &res) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
        res.imageOut,
        res.imageFeatureFilter,
        res.imageColorMap,
        res.imageOut,
    };

    if (res.imageGreyscaleOut != nullptr) {
        imagesToInit.push_back(res.imageGreyscaleOut);
    }

    VulkanResource::initImages(*instance.cmdBufTransfer(), imagesToInit);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = res.imageInput->width,
        .bufferImageHeight = res.imageInput->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
    };
    vk::BufferImageCopy copyColorMapRegion{
        .bufferOffset = 0,
        .bufferRowLength = 256,
        .bufferImageHeight = 1,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{256, 1, 1}
    };
    instance.cmdBufTransfer()->copyBufferToImage(res.stgInput, res.imageInput->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmdBufTransfer()->copyBufferToImage(res.stgRef, res.imageRef->image,  vk::ImageLayout::eGeneral, copyRegion);
    instance.cmdBufTransfer()->copyBufferToImage(res.stgColormap, res.imageColorMap->image,  vk::ImageLayout::eGeneral, copyColorMapRegion);

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

IQM::Bin::FLIPResult IQM::Bin::flip_copy_back(const VulkanInstance &instance, const FLIPResources &res, Timestamps &timestamps, bool colorize) {
    FLIPResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfoCopy);

    vk::BufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = (res.imageInput->width),
        .bufferImageHeight = res.imageInput->height,
        .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = vk::Offset3D{0, 0, 0},
        .imageExtent = vk::Extent3D{res.imageInput->width, res.imageInput->height, 1}
    };

    if (colorize) {
        instance.cmdBufTransfer()->copyImageToBuffer(res.imageInput->image, vk::ImageLayout::eGeneral, res.stgInput, copyRegion);
    } else {
        instance.cmdBufTransfer()->copyImageToBuffer(res.imageGreyscaleOut->image, vk::ImageLayout::eGeneral, res.stgInput, copyRegion);
    }

    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = sizeof(unsigned char) * (res.imageInput->width * res.imageInput->height * 4),
        .size = sizeof(float),
    };
    instance.cmdBufTransfer()->copyBuffer(res.buf, res.stgInput, bufCopy);

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

    std::vector<unsigned char> outputData(res.imageOut->height * res.imageOut->width * 4);
    void * outBufData = res.stgInputMemory.mapMemory(0, ((res.imageOut->height * res.imageOut->width ) + 1) * sizeof(float), {});
    memcpy(outputData.data(), outBufData, res.imageOut->height * res.imageOut->width * 4 * sizeof(unsigned char));
    result.meanFlip = (static_cast<float*>(outBufData))[res.imageOut->height * res.imageOut->width] / (static_cast<float>(res.imageOut->width) * static_cast<float>(res.imageOut->height));

    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    result.imageData = std::move(outputData);

    return result;
}

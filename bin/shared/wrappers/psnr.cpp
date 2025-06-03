/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "psnr.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"
#include "IQM/base/viridis.h"
#include "IQM/base/colorize.h"

using IQM::Bin::InputImage;
using IQM::Bin::VulkanImage;
using IQM::VulkanInstance;
using IQM::Bin::VulkanResource;

IQM::Bin::PSNRResources IQM::Bin::psnr_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance& instance, bool hasOutput, bool colorize) {
    // always 4 channels on input, with 1B per channel, with one float for sum result
    const auto size = (test.width * test.height) * 4 + sizeof(float);
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
    auto [sumBuf, sumMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);
    cmBuf.bindMemory(cmMem, 0);
    sumBuf.bindMemory(sumMem, 0);

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

    return PSNRResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .stgColormap = std::move(cmBuf),
        .stgColormapMemory = std::move(cmMem),
        .sumBuf = std::move(sumBuf),
        .sumMemory = std::move(sumMem),
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

void IQM::Bin::psnr_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches) {
    IQM::PSNR psnr(*instance.device());
    IQM::Colorize colorizer(*instance.device());

    IQM::PSNRVariant variant = IQM::PSNRVariant::Luma;
    if (args.options.contains("--psnr-variant")) {
        auto opt = args.options.at("--psnr-variant");
        if (opt == "luma" || opt == "LUMA") {
            variant = IQM::PSNRVariant::Luma;
        } else if (opt == "rgb" || opt == "RGB") {
            variant = IQM::PSNRVariant::RGB;
        } else if (opt == "yuv" || opt == "YUV") {
            variant = IQM::PSNRVariant::YUV;
        } else {
            throw std::invalid_argument("Unknown PSNR variant");
        }
    }

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

            auto res = psnr_init_res(input, reference, instance, args.outputPath.has_value(), args.colorize);
            timestamps.mark("resources allocated");

            psnr_upload(instance, res, args.outputPath.has_value(), args.colorize);

            auto psnrArgs = IQM::PSNRInput {
                .device = instance.device(),
                .cmdBuf = &*instance.cmdBuf(),
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .bufSum = &res.sumBuf,
                .imgOut = args.outputPath.has_value() ? &res.imageOut->image : nullptr,
                .variant = variant,
                .width = input.width,
                .height = input.height,
            };

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmdBuf()->begin(beginInfo);

            psnr.computeMetric(psnrArgs);

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

            auto result = psnr_copy_back(instance, res, timestamps, args.outputPath.has_value(), args.colorize);

            finishRenderDoc();
            if (match.outPath.has_value()) {
                if (args.colorize) {
                    save_color_image(args.outputPath.value(), result.imageData, input.width, input.height);
                } else {
                    save_char_image(args.outputPath.value(), result.imageData, input.width, input.height);
                }
            }

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.db << " dB" << std::endl;
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

void IQM::Bin::psnr_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::PSNR& psnr, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref) {
    try {
        VulkanResource::resetMemCounter();
        IQM::Colorize colorizer(*instance.device());
        IQM::PSNRVariant variant = IQM::PSNRVariant::Luma;
        if (args.options.contains("--psnr-variant")) {
            auto opt = args.options.at("--psnr-variant");
            if (opt == "luma" || opt == "LUMA") {
                variant = IQM::PSNRVariant::Luma;
            } else if (opt == "rgb" || opt == "RGB") {
                variant = IQM::PSNRVariant::RGB;
            } else if (opt == "yuv" || opt == "YUV") {
                variant = IQM::PSNRVariant::YUV;
            } else {
                throw std::invalid_argument("Unknown PSNR variant");
            }
        }
        Timestamps timestamps;
        auto start = std::chrono::high_resolution_clock::now();

        timestamps.mark("images loaded");

        initRenderDoc();

        auto res = psnr_init_res(input, ref, instance, false, args.colorize);
        timestamps.mark("resources allocated");

        psnr_upload(instance, res, false, args.colorize);

        auto psnrArgs = IQM::PSNRInput {
            .device = instance.device(),
            .cmdBuf = &*instance.cmdBuf(),
            .ivTest = &res.imageInput->imageView,
            .ivRef = &res.imageRef->imageView,
            .bufSum = &res.sumBuf,
            .variant = variant,
            .width = input.width,
            .height = input.height,
        };

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        instance.cmdBuf()->begin(beginInfo);

        psnr.computeMetric(psnrArgs);

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

        auto result = psnr_copy_back(instance, res, timestamps, false, args.colorize);

        finishRenderDoc();

        timestamps.mark("output saved");

        const auto end = std::chrono::high_resolution_clock::now();

        if (args.verbose) {
            std::cout << args.inputPath << ": " << result.db << " dB" << std::endl;
            timestamps.print(start, end);

            double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
            std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to process '" << args.inputPath << "': " << e.what() << std::endl;
    }
}

void IQM::Bin::psnr_upload(const VulkanInstance &instance, const PSNRResources &res, bool hasOutput, bool colorize) {
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

IQM::Bin::PSNRResult IQM::Bin::psnr_copy_back(const VulkanInstance &instance, const PSNRResources &res, Timestamps &timestamps, bool hasOutput, bool colorize) {
    PSNRResult result;

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
    instance.cmdBufTransfer()->copyBuffer(res.sumBuf, res.stgInput, bufCopy);

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

    const auto outBufData = static_cast<float *>(res.stgInputMemory.mapMemory(0, sizeof(float) + res.imageInput->width * res.imageInput->height * 4, {}));
    memcpy(&result.db, outBufData, sizeof(float));

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

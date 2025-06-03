/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <iostream>
#include "fsim.h"
#include "../../shared/debug_utils.h"
#include "../../shared/vulkan_res.h"
#include <IQM/fsim/fft_planner.h>

using IQM::VulkanInstance;

void IQM::Bin::fsim_run(const Args& args, const VulkanInstance& instance, const std::vector<Match>& imageMatches) {
    IQM::FSIM fsim(*instance.device());

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

            auto [dWidth, dHeight] = FSIM::downscaledSize(input.width, input.height);

            auto res = fsim_init_res(input, reference, instance, dWidth, dHeight);
            timestamps.mark("resources allocated");

            fsim_upload(instance, res);

            auto flipInput = IQM::FSIMInput {
                .device = instance.device(),
                .physicalDevice = instance.physicalDevice(),
                .queue = &*instance.queue(),
                .commandPool = &*instance.cmdPool(),
                .cmdBuf = &*instance.cmdBuf(),
                .fenceFft = &res.fftFence,
                .fenceIfft = &res.ifftFence,
                .ivTest = &res.imageInput->imageView,
                .ivRef = &res.imageRef->imageView,
                .ivTestDown = &res.imagesColor[0]->imageView,
                .ivRefDown = &res.imagesColor[1]->imageView,
                .ivTempFloat = {
                    &res.imagesFloat[0]->imageView,
                    &res.imagesFloat[1]->imageView,
                    &res.imagesFloat[2]->imageView,
                    &res.imagesFloat[3]->imageView,
                    &res.imagesFloat[4]->imageView,
                },
                .ivFilterResponsesTest = {
                    &res.imagesRg[0]->imageView,
                    &res.imagesRg[1]->imageView,
                    &res.imagesRg[2]->imageView,
                    &res.imagesRg[3]->imageView,
                },
                .ivFilterResponsesRef = {
                    &res.imagesRg[4]->imageView,
                    &res.imagesRg[5]->imageView,
                    &res.imagesRg[6]->imageView,
                    &res.imagesRg[7]->imageView,
                },
                .ivFinalSums = {
                    &res.imagesFloat[5]->imageView,
                    &res.imagesFloat[6]->imageView,
                    &res.imagesFloat[7]->imageView,
                },
                .imgFinalSums = {
                    &res.imagesFloat[5]->image,
                    &res.imagesFloat[6]->image,
                    &res.imagesFloat[7]->image,
                },
                .bufFft = &res.bufFft,
                .bufIfft = &res.bufIfft,
                .fftApplication = nullptr,
                .fftApplicationInverse = nullptr,
                .width = input.width,
                .height = input.height,
            };

            auto fftApplication = FftPlanner::initForward(flipInput, dWidth, dHeight);
            auto fftApplicationInverse = FftPlanner::initInverse(flipInput, dWidth, dHeight);

            flipInput.fftApplication = &fftApplication;
            flipInput.fftApplicationInverse = &fftApplicationInverse;

            const vk::CommandBufferBeginInfo beginInfo = {
                .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
            };
            instance.cmdBuf()->begin(beginInfo);

            fsim.computeMetric(flipInput);

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

            auto result = fsim_copy_back(instance, res, timestamps);

            finishRenderDoc();

            FftPlanner::destroy(fftApplication);
            FftPlanner::destroy(fftApplicationInverse);

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << match.testPath << ": " << result.fsim << " | " << result.fsimc << std::endl;
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

void IQM::Bin::fsim_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::FSIM &fsim, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref) {
    try {
        VulkanResource::resetMemCounter();
        Timestamps timestamps;
        auto start = std::chrono::high_resolution_clock::now();

        timestamps.mark("images loaded");

        initRenderDoc();

        auto [dWidth, dHeight] = FSIM::downscaledSize(input.width, input.height);

        auto res = fsim_init_res(input, ref, instance, dWidth, dHeight);
        timestamps.mark("resources allocated");

        fsim_upload(instance, res);

        auto fsimInput = IQM::FSIMInput {
            .device = instance.device(),
            .physicalDevice = instance.physicalDevice(),
            .queue = &*instance.queue(),
            .commandPool = &*instance.cmdPool(),
            .cmdBuf = &*instance.cmdBuf(),
            .fenceFft = &res.fftFence,
            .fenceIfft = &res.ifftFence,
            .ivTest = &res.imageInput->imageView,
            .ivRef = &res.imageRef->imageView,
            .ivTestDown = &res.imagesColor[0]->imageView,
            .ivRefDown = &res.imagesColor[1]->imageView,
            .ivTempFloat = {
                &res.imagesFloat[0]->imageView,
                &res.imagesFloat[1]->imageView,
                &res.imagesFloat[2]->imageView,
                &res.imagesFloat[3]->imageView,
                &res.imagesFloat[4]->imageView,
            },
            .ivFilterResponsesTest = {
                &res.imagesRg[0]->imageView,
                &res.imagesRg[1]->imageView,
                &res.imagesRg[2]->imageView,
                &res.imagesRg[3]->imageView,
            },
            .ivFilterResponsesRef = {
                &res.imagesRg[4]->imageView,
                &res.imagesRg[5]->imageView,
                &res.imagesRg[6]->imageView,
                &res.imagesRg[7]->imageView,
            },
            .ivFinalSums = {
                &res.imagesFloat[5]->imageView,
                &res.imagesFloat[6]->imageView,
                &res.imagesFloat[7]->imageView,
            },
            .imgFinalSums = {
                &res.imagesFloat[5]->image,
                &res.imagesFloat[6]->image,
                &res.imagesFloat[7]->image,
            },
            .bufFft = &res.bufFft,
            .bufIfft = &res.bufIfft,
            .fftApplication = nullptr,
            .fftApplicationInverse = nullptr,
            .width = input.width,
            .height = input.height,
        };

        auto fftApplication = FftPlanner::initForward(fsimInput, dWidth, dHeight);
        auto fftApplicationInverse = FftPlanner::initInverse(fsimInput, dWidth, dHeight);

        fsimInput.fftApplication = &fftApplication;
        fsimInput.fftApplicationInverse = &fftApplicationInverse;

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
        };
        instance.cmdBuf()->begin(beginInfo);

        fsim.computeMetric(fsimInput);

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

        auto result = fsim_copy_back(instance, res, timestamps);

        finishRenderDoc();

        FftPlanner::destroy(fftApplication);
        FftPlanner::destroy(fftApplicationInverse);

        const auto end = std::chrono::high_resolution_clock::now();
        if (args.verbose) {
            std::cout << args.inputPath << ": " << result.fsim << " | " << result.fsimc << std::endl;
            timestamps.print(start, end);

            double mbSize = static_cast<double>(VulkanResource::memCounter()) / 1024 / 1024;
            std::cout << "VRAM used for resources: " << mbSize << " MB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to process '" << args.inputPath << "': " << e.what() << std::endl;
    }
}

IQM::Bin::FSIMResources IQM::Bin::fsim_init_res(const InputImage &test, const InputImage &ref, const VulkanInstance& instance, const unsigned dWidth, const unsigned dHeight) {
    // always 4 channels on input, with 1B per channel
    const auto inputSize = (test.width * test.height) * 4;

    // input buffers
    auto [stgBuf, stgMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        inputSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached
    );
    auto [stgRefBuf, stgRefMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        inputSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    stgBuf.bindMemory(stgMem, 0);
    stgRefBuf.bindMemory(stgRefMem, 0);

    void * inBufData = stgMem.mapMemory(0, inputSize, {});
    memcpy(inBufData, test.data.data(), inputSize);
    stgMem.unmapMemory();

    inBufData = stgRefMem.mapMemory(0, inputSize, {});
    memcpy(inBufData, ref.data.data(), inputSize);
    stgRefMem.unmapMemory();

    // rest of buffers
    const auto fftSize = sizeof(float) * (dWidth * dHeight) * 2 * 2;
    const auto ifftSize = sizeof(float) * (dWidth * dHeight) * 2 * FSIM_ORIENTATIONS * FSIM_SCALES * 3;

    auto [fftBuf, fftMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        fftSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    fftBuf.bindMemory(fftMem, 0);
    auto [ifftBuf, ifftMem] = VulkanResource::createBuffer(
        *instance.device(),
        *instance.physicalDevice(),
        ifftSize,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    ifftBuf.bindMemory(ifftMem, 0);

    // images
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

    vk::ImageCreateInfo floatImageInfo {srcImageInfo};
    floatImageInfo.format = vk::Format::eR32Sfloat;
    floatImageInfo.extent = vk::Extent3D(dWidth, dHeight, 1),
    floatImageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;

    vk::ImageCreateInfo rgImageInfo = {floatImageInfo};
    rgImageInfo.format = vk::Format::eR32G32Sfloat;

    vk::ImageCreateInfo colorImageInfo = {floatImageInfo};
    colorImageInfo.format = vk::Format::eR32G32B32A32Sfloat;

    auto const imageInput = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));
    auto const imageRef = std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), srcImageInfo));

    auto imagesFloat = std::vector<std::shared_ptr<VulkanImage>>();
    auto imagesRg = std::vector<std::shared_ptr<VulkanImage>>();
    auto imagesColor = std::vector<std::shared_ptr<VulkanImage>>();

    for (int i = 0; i < 8; i++) {
        imagesFloat.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), floatImageInfo)));
    }

    for (int i = 0; i < 8; i++) {
        imagesRg.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), rgImageInfo)));
    }

    for (int i = 0; i < 2; i++) {
        imagesColor.emplace_back(std::make_shared<VulkanImage>(VulkanResource::createImage(*instance.device(), *instance.physicalDevice(), colorImageInfo)));
    }

    return FSIMResources{
        .stgInput = std::move(stgBuf),
        .stgInputMemory = std::move(stgMem),
        .stgRef = std::move(stgRefBuf),
        .stgRefMemory = std::move(stgRefMem),
        .imageInput = imageInput,
        .imageRef = imageRef,
        .imagesFloat = imagesFloat,
        .imagesRg = imagesRg,
        .imagesColor = imagesColor,
        .bufFft = std::move(fftBuf),
        .memFft = std::move(fftMem),
        .bufIfft = std::move(ifftBuf),
        .memIfft = std::move(ifftMem),
        .uploadDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .computeDone = instance.device()->createSemaphore(vk::SemaphoreCreateInfo{}),
        .transferFence = instance.device()->createFence(vk::FenceCreateInfo{}),
        .fftFence = instance.device()->createFence(vk::FenceCreateInfo{}),
        .ifftFence = instance.device()->createFence(vk::FenceCreateInfo{}),
    };
}

void IQM::Bin::fsim_upload(const VulkanInstance& instance, const FSIMResources& res) {
    const vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfo);

    std::vector imagesToInit = {
        res.imageInput,
        res.imageRef,
    };

    imagesToInit.insert(imagesToInit.end(), res.imagesFloat.begin(), res.imagesFloat.end());
    imagesToInit.insert(imagesToInit.end(), res.imagesRg.begin(), res.imagesRg.end());
    imagesToInit.insert(imagesToInit.end(), res.imagesColor.begin(), res.imagesColor.end());

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

IQM::Bin::FSIMResult IQM::Bin::fsim_copy_back(const VulkanInstance& instance, const FSIMResources& res, Timestamps &timestamps) {
    FSIMResult result;

    // copy out
    const vk::CommandBufferBeginInfo beginInfoCopy = {
        .flags = vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
    };
    instance.cmdBufTransfer()->begin(beginInfoCopy);

    vk::BufferCopy bufCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = 3 * sizeof(float),
    };
    instance.cmdBufTransfer()->copyBuffer(res.bufFft, res.stgInput, bufCopy);

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

    std::vector<float> outputData(3);
    void * outBufData = res.stgInputMemory.mapMemory(0, 3 * sizeof(float), {});
    memcpy(outputData.data(), outBufData, 3 * sizeof(float));

    res.stgInputMemory.unmapMemory();
    timestamps.mark("end copy from GPU");

    result.fsim = outputData[1] / outputData[0];
    result.fsimc = outputData[2] / outputData[0];

    return result;
}
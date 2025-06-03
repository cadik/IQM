/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_SVD_H
#define IQM_BIN_SVD_H

#include <IQM/svd.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct SVDResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer svdBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory svdMemory = VK_NULL_HANDLE;
        vk::raii::Buffer reduceBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory reduceMemory = VK_NULL_HANDLE;
        vk::raii::Buffer sortBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory sortMemory = VK_NULL_HANDLE;
        vk::raii::Buffer sortTempBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory sortTempMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 2x R f32 converted images
        std::vector<std::shared_ptr<VulkanImage>> imagesFloat;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct SVDResult {
        std::vector<float> imageData;
        float msvd;
    };

    void svd_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void svd_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::SVD& svd, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref);

    SVDResources svd_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance);
    void svd_upload(const IQM::VulkanInstance& instance, const SVDResources& res);
    SVDResult svd_copy_back(const IQM::VulkanInstance& instance, const SVDResources& res, Timestamps &timestamps, uint32_t pixelCount);
}

#endif //IQM_BIN_SVD_H

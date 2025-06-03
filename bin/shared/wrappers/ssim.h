/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_SSIM_H
#define IQM_BIN_SSIM_H

#include <IQM/ssim.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct SSIMResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgColormap = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgColormapMemory = VK_NULL_HANDLE;
        vk::raii::Buffer mssimBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory mssimMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 5x R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesBlurred;

        // R f32 output image
        std::shared_ptr<VulkanImage> imageOut;

        // R u8 export image
        std::shared_ptr<VulkanImage> imageExport;

        // RGBA f32 colormap
        std::shared_ptr<VulkanImage> imageColorMap;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct SSIMResult {
        std::vector<unsigned char> imageData;
        float mssim;
    };

    void ssim_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void ssim_run_single(const IQM::ProfileArgs &args, const IQM::VulkanInstance &instance, IQM::SSIM& ssim, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref);

    SSIMResources ssim_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance);
    void ssim_upload(const IQM::VulkanInstance& instance, const SSIMResources& res);
    SSIMResult ssim_copy_back(const IQM::VulkanInstance& instance, const SSIMResources& res, Timestamps &timestamps, uint32_t kernelSize, bool colorize);
}

#endif //IQM_BIN_SSIM_H

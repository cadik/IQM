/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_FSIM_H
#define IQM_BIN_FSIM_H

#include <IQM/fsim.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct FSIMResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // 8x R f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesFloat;
        // 8x RG f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesRg;
        // 2x RGBA f32 intermediate images
        std::vector<std::shared_ptr<VulkanImage>> imagesColor;

        vk::raii::Buffer bufFft = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memFft = VK_NULL_HANDLE;
        vk::raii::Buffer bufIfft = VK_NULL_HANDLE;
        vk::raii::DeviceMemory memIfft = VK_NULL_HANDLE;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;

        // FFT
        vk::raii::Fence fftFence = VK_NULL_HANDLE;
        vk::raii::Fence ifftFence = VK_NULL_HANDLE;
    };

    struct FSIMResult {
        float fsim;
        float fsimc;
    };

    void fsim_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void fsim_run_single(const IQM::ProfileArgs& args, const IQM::VulkanInstance& instance, IQM::FSIM& fsim, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref);

    FSIMResources fsim_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance, unsigned dWidth, unsigned dHeight);
    void fsim_upload(const IQM::VulkanInstance& instance, const FSIMResources& res);
    FSIMResult fsim_copy_back(const IQM::VulkanInstance& instance, const FSIMResources& res, Timestamps &timestamps);
}

#endif //IQM_BIN_FSIM_H

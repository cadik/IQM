/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_LPIPS_H
#define IQM_BIN_LPIPS_H

#include <IQM/lpips.h>
#include "../../shared/vulkan.h"
#include "../../shared/vulkan_res.h"
#include "../../shared/io.h"
#include "../../shared/timestamps.h"
#include "../../IQM/args.h"
#include "../../IQM-profile/args.h"

namespace IQM::Bin {
    struct LPIPSResources {
        vk::raii::Buffer stgInput = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgRef = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer stgColormap = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgColormapMemory = VK_NULL_HANDLE;

        vk::raii::Buffer convInputBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory convInputMemory = VK_NULL_HANDLE;
        vk::raii::Buffer convRefBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory convRefMemory = VK_NULL_HANDLE;
        vk::raii::Buffer compareBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory compareMemory = VK_NULL_HANDLE;

        // RGBA u8 input images
        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        // optional R f32 output image
        std::shared_ptr<VulkanImage> imageOut;

        // optional u8 export image
        std::shared_ptr<VulkanImage> imageExport;

        // RGBA f32 colormap
        std::shared_ptr<VulkanImage> imageColorMap;

        vk::raii::Semaphore uploadDone = VK_NULL_HANDLE;
        vk::raii::Semaphore computeDone = VK_NULL_HANDLE;
        vk::raii::Fence transferFence = VK_NULL_HANDLE;
    };

    struct LPIPSModelResources {
        vk::raii::Buffer stgWeights = VK_NULL_HANDLE;
        vk::raii::DeviceMemory stgWeightsMemory = VK_NULL_HANDLE;
        vk::raii::Buffer weightsBuf = VK_NULL_HANDLE;
        vk::raii::DeviceMemory weightsMemory = VK_NULL_HANDLE;
    };

    struct LPIPSResult {
        std::vector<unsigned char> imageData;
        float distance;
    };

    void lpips_run(const IQM::Bin::Args& args, const IQM::VulkanInstance& instance, const std::vector<Match>& imageMatches);
    void lpips_run_single(const IQM::ProfileArgs& args, const IQM::VulkanInstance& instance, IQM::LPIPS& lpips, const IQM::Bin::InputImage& input, const IQM::Bin::InputImage& ref, const std::vector<float> &lpipsModel);

    LPIPSResources lpips_init_res(const InputImage &test, const InputImage &ref, const IQM::VulkanInstance& instance, const LPIPSBufferSizes &bufferSizes, bool hasOutput, bool colorize);
    void lpips_upload(const IQM::VulkanInstance& instance, const LPIPSResources& res, const LPIPSModelResources &model, unsigned long modelSize, bool hasOutput, bool colorize);
    LPIPSModelResources lpips_load_model(const IQM::VulkanInstance& instance, unsigned long modelSize, const std::vector<float> &modelData);
    LPIPSResult lpips_copy_back(const IQM::VulkanInstance& instance, const LPIPSResources& res, Timestamps &timestamps, bool hasOutput, bool colorize);
}

#endif //IQM_BIN_LPIPS_H

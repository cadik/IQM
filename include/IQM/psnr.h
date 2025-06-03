/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_PSNR_H
#define IQM_PSNR_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    enum PSNRVariant {
        Luma = 0,
        RGB = 1,
        YUV = 2,
    };

    /**
     * Input parameters for PSNR computation.
     *
     * Source image views `ivTest` and `ivRef` are expected to be views into RGBA u8 images of WxH.
     * All images should be in layout GENERAL.
     *
     * Buffer should have size of WxHx4 bytes.
     *
     * If the image `imgOut` is not null, square errors will be saved there.
     *
     * PSNR result is on the zero-th index of `bufSum`.
     */
    struct PSNRInput {
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef;
        const vk::raii::Buffer *bufSum;
        const vk::raii::Image *imgOut = nullptr;
        PSNRVariant variant;
        unsigned width, height;
    };

    class PSNR {
    public:
        explicit PSNR(const vk::raii::Device &device);
        void computeMetric(const PSNRInput& input);
    private:
        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutPack = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelinePack = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutPack = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetPack = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSum = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSum = VK_NULL_HANDLE;

        vk::raii::Pipeline pipelinePost = VK_NULL_HANDLE;

        void initDescriptors(const PSNRInput& input);
    };
}

#endif //IQM_PSNR_H

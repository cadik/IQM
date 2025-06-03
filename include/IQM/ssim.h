/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef SSIM_H
#define SSIM_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    /**
     * Input parameters for SSIM computation.
     *
     * Source image views `ivTest` and `ivRef` are expected to be views into RGBA u8 images of WxH.
     * Rest of image views are expected to be in R f32 format with dimensions WxH.
     * All images should be in layout GENERAL.
     *
     * Buffer should have size of WxHx4 bytes.
     *
     * After the computation the resulting graphical measure is in `imgOut`.
     * MSSIM result is on the zero-th index of `bufMssim`.
     */
    struct SSIMInput {
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef, *ivMeanTest, *ivMeanRef, *ivVarTest, *ivVarRef, *ivCovar, *ivOut;
        const vk::raii::Image *imgOut;
        const vk::raii::Buffer *bufMssim;
        unsigned width, height;
    };

    class SSIM {
    public:
        explicit SSIM(const vk::raii::Device &device);
        void computeMetric(const SSIMInput& input);

        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
        float sigma = 1.5;
    private:
        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSsim = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSsim = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSsim = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSsim = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutLumapack = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineLumapack = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutLumapack = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetLumapack = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutGauss = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGauss = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGaussHorizontal = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGauss = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutMssim = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineMssim = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutMssim = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetMssim = VK_NULL_HANDLE;

        void initDescriptors(const SSIMInput& input);
    };
}

#endif //SSIM_H

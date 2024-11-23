/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_H
#define FSIM_H
#include <opencv2/core/mat.hpp>
#include <vkFFT.h>

#include "../../timestamps.h"
#include "../base/vulkan_runtime.h"

namespace IQM::GPU {
    struct FSIMResult {
        cv::Mat image;
        float fsim;
        float fsimc;
        Timestamps timestamps;
    };

    class FSIM {
    public:
        explicit FSIM(const VulkanRuntime &runtime);
        FSIMResult computeMetric(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);

        bool doColorComparison = true;
        int orientations = 4;
        int scales = 4;

    private:
        static int computeDownscaleFactor(int cols, int rows);
        void sendImagesToGpu(const VulkanRuntime &runtime, const cv::Mat &image, const cv::Mat &ref);
        void createDownscaledImages(const VulkanRuntime & runtime, int width_downscale, int height_downscale);
        void computeDownscaledImages(const VulkanRuntime & runtime, int, int, int);
        void createLowpassFilter(const VulkanRuntime & runtime, int, int);
        void createGradientMap(const VulkanRuntime & runtime, int, int);
        void computeFft(const VulkanRuntime &runtime, FSIMResult &res, int width, int height);

        vk::raii::ShaderModule downscaleKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutDownscale = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineDownscale = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleRef = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        std::shared_ptr<VulkanImage> imageInputDownscaled;
        std::shared_ptr<VulkanImage> imageRefDownscaled;
        std::shared_ptr<VulkanImage> imageLowpassFilter;

        vk::raii::ShaderModule lowpassFilterKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutLowpassFilter = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineLowpassFilter = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetLowpassFilter = VK_NULL_HANDLE;

        // gradient map pass
        vk::raii::PipelineLayout layoutGradientMap = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGradientMap = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGradientMapIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGradientMapRef = VK_NULL_HANDLE;
        vk::raii::ShaderModule kernelGradientMap = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageGradientMapInput;
        std::shared_ptr<VulkanImage> imageGradientMapRef;

        // extract luma for FFT library pass
        vk::raii::PipelineLayout layoutExtractLuma = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineExtractLuma = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetExtractLumaIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetExtractLumaRef = VK_NULL_HANDLE;
        vk::raii::ShaderModule kernelExtractLuma = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageFftInput;
        std::shared_ptr<VulkanImage> imageFftRef;
    };
}

#endif //FSIM_H

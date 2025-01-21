/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_H
#define FSIM_H
#include <vkFFT.h>

#include "../../input_image.h"
#include "../../timestamps.h"
#include "../base/vulkan_runtime.h"
#include "steps/fsim_log_gabor.h"
#include "steps/fsim_lowpass_filter.h"
#include "steps/fsim_angular_filter.h"
#include "steps/fsim_estimate_energy.h"
#include "steps/fsim_filter_combinations.h"
#include "steps/fsim_final_multiply.h"
#include "steps/fsim_noise_power.h"
#include "steps/fsim_phase_congruency.h"
#include "steps/fsim_sum_filter_responses.h"

namespace IQM::GPU {
    constexpr int FSIM_ORIENTATIONS = 4;
    constexpr int FSIM_SCALES = 4;

    struct FSIMResult {
        float fsim;
        float fsimc;
        Timestamps timestamps;
    };

    class FSIM {
    public:
        explicit FSIM(const VulkanRuntime &runtime);
        FSIMResult computeMetric(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);

    private:
        static int computeDownscaleFactor(int width, int height);
        void sendImagesToGpu(const VulkanRuntime &runtime, const InputImage &image, const InputImage &ref);
        void createDownscaledImages(const VulkanRuntime & runtime, int width_downscale, int height_downscale);
        void computeDownscaledImages(const VulkanRuntime & runtime, int, int, int);
        void createGradientMap(const VulkanRuntime & runtime, int, int);
        void initFftLibrary(const VulkanRuntime &runtime, int width, int height);
        void teardownFftLibrary();
        void computeFft(const VulkanRuntime &runtime, int width, int height);
        void computeMassInverseFft(const VulkanRuntime & runtime, const vk::raii::Buffer &buffer);

        FSIMLowpassFilter lowpassFilter;
        FSIMLogGabor logGaborFilter;
        FSIMAngularFilter angularFilter;
        FSIMFilterCombinations combinations;
        FSIMSumFilterResponses sumFilterResponses;
        FSIMNoisePower noise_power;
        FSIMEstimateEnergy estimateEnergy;
        FSIMPhaseCongruency phaseCongruency;
        FSIMFinalMultiply final_multiply;

        vk::raii::ShaderModule downscaleKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutDownscale = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineDownscale = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscaleRef = VK_NULL_HANDLE;

        std::shared_ptr<VulkanImage> imageInput;
        std::shared_ptr<VulkanImage> imageRef;

        std::shared_ptr<VulkanImage> imageInputDownscaled;
        std::shared_ptr<VulkanImage> imageRefDownscaled;

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

        vk::raii::DeviceMemory memoryFft = VK_NULL_HANDLE;
        vk::raii::Buffer bufferFft = VK_NULL_HANDLE;

        // FFT lib
        VkFFTApplication fftApplication{};
        VkFFTApplication fftApplicationInverse{};
        vk::raii::Fence fftFence = VK_NULL_HANDLE;
        vk::raii::Fence fftFenceInverse = VK_NULL_HANDLE;
    };
}

#endif //FSIM_H

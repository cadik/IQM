/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_H
#define FSIM_H
#include <vkFFT.h>

#include <IQM/base/vulkan_runtime.h>
#include <IQM/fsim/log_gabor.h>
#include <IQM/fsim/angular_filter.h>
#include <IQM/fsim/estimate_energy.h>
#include <IQM/fsim/filter_combinations.h>
#include <IQM/fsim/final_multiply.h>
#include <IQM/fsim/noise_power.h>
#include <IQM/fsim/phase_congruency.h>
#include <IQM/fsim/sum_filter_responses.h>
#include <IQM/fsim/partitions.h>

namespace IQM {
    constexpr int FSIM_ORIENTATIONS = 4;
    constexpr int FSIM_SCALES = 4;

    /**
     * ## Images with format RGBA u8 | (WxH)
     * - *ivTest, *ivRef
     * ## Images with format RGBA u8 | D(WxH)
     * - *ivTestDown, *ivRefDown
     * ## Images with format R f32 | D(WxH)
     * - *ivFinalSums[3], *ivTempFloat[5]
     * ## Images with format RG f32 | D(WxH)
     * - *ivFilterResponsesTest[4]
     * - *ivFilterResponsesRef[4]
     *
     * Both supplied buffers are primarily used for FFT computation,
     * but after that are reused for other work, such as parallel sums or sorts.
     *
     * `bufFft` must have size D(WxH) x sizeof(float) x 4
     * `bufIfft` must have size D(WxH) x sizeof(float) x 96
     *
     * After finishing, output values FSIM and FSIM can be computed from values in `bufFft`:
     *  - FSIM = bufFft[1] / bufFft[0];
     *  - FSIMc = bufFft[2] / bufFft[0];
     */
    struct FSIMInput {
        const vk::raii::Device *device;
        const vk::raii::PhysicalDevice *physicalDevice;
        const vk::raii::Queue *queue;
        const vk::raii::CommandPool *commandPool;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::Fence *fenceFft, *fenceIfft;
        const vk::raii::ImageView *ivTest, *ivRef, *ivTestDown, *ivRefDown;
        const vk::raii::ImageView *ivTempFloat[5];
        const vk::raii::ImageView *ivFilterResponsesTest[FSIM_ORIENTATIONS];
        const vk::raii::ImageView *ivFilterResponsesRef[FSIM_ORIENTATIONS];
        const vk::raii::ImageView *ivFinalSums[3];
        const vk::raii::Image *imgFinalSums[3];
        const vk::raii::Buffer *bufFft, *bufIfft;
        // FFT lib
        VkFFTApplication *fftApplication;
        VkFFTApplication *fftApplicationInverse;
        unsigned width, height;
    };

    class FSIM {
    public:
        explicit FSIM(const vk::raii::Device &device);
        void computeMetric(const FSIMInput& input);

        static std::pair<unsigned, unsigned> downscaledSize(unsigned width, unsigned height);

    private:
        void initDescriptors(const FSIMInput& input, const FftBufferPartitions& partitions);
        static int computeDownscaleFactor(int width, int height);
        void computeDownscaledImages(const FSIMInput& input, int factor, int width, int height);
        void createGradientMap(const FSIMInput& input, int, int);
        void computeFft(const FSIMInput& input, unsigned width, unsigned height);
        void computeMassInverseFft(const FSIMInput& input);

        static unsigned sortBufSize(unsigned dWidth, unsigned dHeight);

        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::DescriptorSetLayout descSetLayoutImageOp = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutImBufOp = VK_NULL_HANDLE;

        FSIMLogGabor logGaborFilter;
        FSIMAngularFilter angularFilter;
        FSIMFilterCombinations combinations;
        FSIMSumFilterResponses sumFilterResponses;
        FSIMNoisePower noise_power;
        FSIMEstimateEnergy estimateEnergy;
        FSIMPhaseCongruency phaseCongruency;
        FSIMFinalMultiply final_multiply;

        vk::raii::PipelineLayout layoutDownscale = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineDownscale = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetDownscale = VK_NULL_HANDLE;

        // gradient map pass
        vk::raii::PipelineLayout layoutGradientMap = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineGradientMap = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetGradientMap = VK_NULL_HANDLE;

        // extract luma for FFT library pass
        vk::raii::PipelineLayout layoutExtractLuma = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineExtractLuma = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetExtractLumaIn = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetExtractLumaRef = VK_NULL_HANDLE;
    };
}

#endif //FSIM_H

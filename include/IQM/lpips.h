/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef LPIPS_H
#define LPIPS_H
#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct LPIPSInput {
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef;
        const vk::raii::Image *imgOut;
        const vk::raii::Buffer *bufWeights, *bufTest, *bufRef, *bufComp;
        unsigned width, height;
    };

    struct ConvParams {
        unsigned kernelSize;
        unsigned inChannels;
        unsigned outChannels;
        unsigned padding = 1;
        unsigned stride = 1;
    };

    struct ConvBufferHalves {
        unsigned long input;
        unsigned long conv;
    };

    struct LPIPSBufferSizes {
        unsigned long bufTest;
        unsigned long bufRef;
        unsigned long bufComp;
        unsigned long bufWeights;
    };

    class LPIPS {
    public:
        explicit LPIPS(const vk::raii::Device &device);
        [[nodiscard]] unsigned long modelSize() const;
        [[nodiscard]] LPIPSBufferSizes bufferSizes(unsigned width, unsigned height) const;
        void computeMetric(const LPIPSInput& input);

        const ConvParams blocks[5] = {
            {
                .kernelSize = 11,
                .inChannels = 3,
                .outChannels = 64,
                .padding = 2,
                .stride = 4
            },
            {
                .kernelSize = 5,
                .inChannels = 64,
                .outChannels = 192,
                .padding = 2
            },
            {
                .kernelSize = 3,
                .inChannels = 192,
                .outChannels = 384,
                .padding = 1,
            },
            {
                .kernelSize = 3,
                .inChannels = 384,
                .outChannels = 256,
                .padding = 1,
            },
            {
                .kernelSize = 3,
                .inChannels = 256,
                .outChannels = 256,
                .padding = 1,
            }
        };

    private:
        void createConvPipelines(const vk::raii::Device &device, const vk::raii::ShaderModule &sm, const vk::raii::ShaderModule &smBig, const vk::raii::PipelineLayout &layout);

        void setUpDescriptors(const LPIPSInput& input) const;
        void preprocess(const LPIPSInput& input);
        void conv0(const LPIPSInput& input);
        void conv1(const LPIPSInput& input);
        void conv2(const LPIPSInput& input);
        void conv3(const LPIPSInput& input);
        void conv4(const LPIPSInput& input);
        void reconstruct(const LPIPSInput& input);
        void average(const LPIPSInput& input);

        [[nodiscard]] ConvBufferHalves bufferHalves(unsigned width, unsigned height) const;

        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout preprocessLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline preprocessPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout preprocessDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet preprocessDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout convLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline convPipelineBig = VK_NULL_HANDLE;
        vk::raii::Pipeline convPipelineMedium = VK_NULL_HANDLE;
        vk::raii::Pipeline convPipelineSmall = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout convDescSetLayout = VK_NULL_HANDLE;

        std::vector<vk::raii::DescriptorSet> convDescSets;

        vk::raii::PipelineLayout maxPoolLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline maxPoolPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout maxPoolDescSetLayout = VK_NULL_HANDLE;
        std::vector<vk::raii::DescriptorSet> maxPoolDescSets;

        vk::raii::PipelineLayout compareLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline comparePipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout compareDescSetLayout = VK_NULL_HANDLE;
        std::vector<vk::raii::DescriptorSet> compareDescSets;

        vk::raii::PipelineLayout reconstructLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline reconstructPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet reconstructDescSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::Pipeline postprocessPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
    };
}

#endif //LPIPS_H

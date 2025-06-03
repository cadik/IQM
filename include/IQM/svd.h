/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef SVD_H
#define SVD_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct SVDInput {
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        const vk::raii::ImageView *ivTest, *ivRef, *ivConvTest, *ivConvRef;
        const vk::raii::Buffer *bufSvd;
        const vk::raii::Buffer *bufReduce;
        const vk::raii::Buffer *bufSort;
        const vk::raii::Buffer *bufSortTemp;
        unsigned width, height;
    };

    class SVD {
    public:
        explicit SVD(const vk::raii::Device &device);
        void computeMetric(const SVDInput& input);

    private:
        void convertColorSpace(const SVDInput& input);
        void computeSvd(const SVDInput& input);
        void reduceSingularValues(const SVDInput& input);
        void sortBlocks(const SVDInput& input);
        void computeMsvd(const SVDInput& input);

        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutConvert = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineConvert = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutConvert = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetConvert = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSvd = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSvd = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSvd = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSvd = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutReduce = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineReduce = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutReduce = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetReduce = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSort = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSortHistogram = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSort = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSortHistogram = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSort = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortOdd = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramOdd = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSum = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSum = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSum = VK_NULL_HANDLE;

        void initDescriptors(const SVDInput& input);
    };
}

#endif //SVD_H

/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef FSIM_NOISE_POWER_H
#define FSIM_NOISE_POWER_H

#include <IQM/base/vulkan_runtime.h>
#include <IQM/fsim/partitions.h>

namespace IQM {
    struct FSIMInput;

    class FSIMNoisePower {
        friend class FSIM;
        explicit FSIMNoisePower(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool);
        void setUpDescriptors(const FSIMInput& input, unsigned width, unsigned height, const FftBufferPartitions& partitions) const;
        void computeNoisePower(const FSIMInput &input, unsigned width, unsigned height);

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutSort = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layoutSortHistogram = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSort = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineSortHistogram = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutSort = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortOdd = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramEven = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetSortHistogramOdd = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layoutNoisePower = VK_NULL_HANDLE;
        vk::raii::Pipeline pipelineNoisePower = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayoutNoisePower = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSetNoisePower = VK_NULL_HANDLE;
    };
}

#endif //FSIM_NOISE_POWER_H

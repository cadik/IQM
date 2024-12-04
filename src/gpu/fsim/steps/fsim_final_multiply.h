/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef FSIM_FINAL_MULTIPLY_H
#define FSIM_FINAL_MULTIPLY_H
#include "../../base/vulkan_runtime.h"


namespace IQM::GPU {
    class FSIMFinalMultiply {
    public:
        explicit FSIMFinalMultiply(const VulkanRuntime& runtime);
        std::pair<float, float> computeMetrics(
            const VulkanRuntime &runtime,
            const std::vector<std::shared_ptr<VulkanImage>> &inputImgs,
            const std::vector<std::shared_ptr<VulkanImage>> &gradientImgs,
            const std::vector<std::shared_ptr<VulkanImage>> &pcImgs,
            int width,
            int height
        );

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        std::vector<std::shared_ptr<VulkanImage>> images;

        vk::raii::ShaderModule sumKernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout sumLayout = VK_NULL_HANDLE;
        vk::raii::Pipeline sumPipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout sumDescSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet sumDescSet = VK_NULL_HANDLE;

        vk::raii::Buffer sumBuffer = VK_NULL_HANDLE;
        vk::raii::DeviceMemory sumMemory = VK_NULL_HANDLE;
    private:
        void prepareImageStorage(
            const VulkanRuntime &runtime,
            const std::vector<std::shared_ptr<VulkanImage>> &inputImgs,
            const std::vector<std::shared_ptr<VulkanImage>> &gradientImgs,
            const std::vector<std::shared_ptr<VulkanImage>> &pcImgs,
            int width,
            int height
        );
        std::pair<float, float> sumImages(const VulkanRuntime & runtime, int width, int height);
    };
}

#endif //FSIM_FINAL_MULTIPLY_H

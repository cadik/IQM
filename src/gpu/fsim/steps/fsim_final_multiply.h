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
        std::pair<float, float> computeMetrics(const VulkanRuntime& runtime, int width, int height);

        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;

        std::array<std::shared_ptr<VulkanImage>, 3> images;
    private:
        void prepareImageStorage(const VulkanRuntime & runtime, int width, int height);
    };
}

#endif //FSIM_FINAL_MULTIPLY_H

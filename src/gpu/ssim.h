#ifndef SSIM_H
#define SSIM_H
#include "base/vulkan_runtime.h"

namespace IQM::GPU {
    class SSIM {
    public:
        explicit SSIM(const VulkanRuntime &runtime);
        void computeMetric(const VulkanRuntime &runtime);

        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
    private:
        vk::raii::ShaderModule kernel = VK_NULL_HANDLE;
        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
    };
}

#endif //SSIM_H

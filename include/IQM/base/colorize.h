/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */
#ifndef IQM_COLORIZE_H
#define IQM_COLORIZE_H

#include <IQM/base/vulkan_runtime.h>

namespace IQM {
    struct ColorizeInput {
        const vk::raii::Device *device;
        const vk::raii::CommandBuffer *cmdBuf;
        // Input image, expected in R f32 format
        const vk::raii::ImageView *ivIn;
        // Output image, expected in RGBA u8 format
        const vk::raii::ImageView *ivOut;
        // Colormap image, expected in RGBA f32 format
        const vk::raii::ImageView *ivColormap;
        // Should the colormap be inverted
        bool invert = false;
        // Scale to make small errors more visible
        float scaler = 1.0f;
        unsigned width, height;
    };

    /**
     * Universal class for post-processing created images.
     * Can be used with any method which returns greyscale image.
     * Takes the greyscale image, and maps it to color image with given map.
     */
    class Colorize {
    public:
        explicit Colorize(const vk::raii::Device &device);
        void compute(const ColorizeInput& input);
    private:
        vk::raii::DescriptorPool descPool = VK_NULL_HANDLE;

        vk::raii::PipelineLayout layout = VK_NULL_HANDLE;
        vk::raii::Pipeline pipeline = VK_NULL_HANDLE;
        vk::raii::DescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
        vk::raii::DescriptorSet descSet = VK_NULL_HANDLE;
    };
}

#endif //IQM_COLORIZE_H

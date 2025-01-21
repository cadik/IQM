/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "args.h"
#include "gpu/base/vulkan_runtime.h"
#include "debug_utils.h"
#include "input_image.h"

#if COMPILE_SSIM
#include <ssim.h>
#endif

#if COMPILE_SVD
#include <svd.h>
#endif

#if COMPILE_FSIM
#include <fsim.h>
#endif

#if COMPILE_FLIP
#include <flip.h>
#endif

InputImage load_image(const std::string &filename) {
    // force all images to always open in RGBA format to prevent issues with separate RGB and RGBA loading
    int x, y, channels;
    unsigned char* data = stbi_load(filename.c_str(), &x, &y, &channels, 4);
    if (data == nullptr) {
        const auto err = stbi_failure_reason();
        const auto msg = std::string("Failed to load image '" + filename + "', reason: " + err);
        throw std::runtime_error(msg);
    }

    std::vector<unsigned char> dataVec(x * y * 4);
    memcpy(dataVec.data(), data, x * y * 4 * sizeof(char));

    stbi_image_free(data);

    return InputImage{
        .width = x,
        .height = y,
        .data = std::move(dataVec)
    };
}

std::vector<unsigned char> convertFloatToChar(const std::vector<float>& data) {
    std::vector<unsigned char> result(data.size());

    for (int i = 0; i < data.size(); ++i) {
        result[i] = static_cast<unsigned char>(data[i] * 255.0f);
    }

    return result;
}

void ssim(const IQM::Args& args) {
#ifdef COMPILE_SSIM
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    const IQM::GPU::VulkanRuntime vulkan;

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    IQM::GPU::SSIM ssim(vulkan);

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = ssim.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "MSSIM: " << result.mssim << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        auto converted = convertFloatToChar(result.imageData);

        auto saveResult = stbi_write_png(args.outputPath.value().c_str(), result.width, result.height, 1, converted.data(), result.width * sizeof(unsigned char));
        if (saveResult == 0) {
            throw std::runtime_error("Failed to save output image");
        }
    }
#else
    throw std::runtime_error("SSIM support was not compiled");
#endif
}

void cw_ssim_ref(const IQM::Args& args) {
    /*auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    auto method = IQM::CPU::CW_SSIM_Ref();

    const auto start = std::chrono::high_resolution_clock::now();
    method.computeMetric(image, ref);
    const auto end = std::chrono::high_resolution_clock::now();

    if (args.verbose) {
        auto execTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << execTime << std::endl;
    }*/
}

void svd(const IQM::Args& args) {
#ifdef COMPILE_SVD
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    const IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::SVD svd(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = svd.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "M-SVD: " << result.msvd << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }

    if (args.outputPath.has_value()) {
        auto converted = convertFloatToChar(result.imageData);

        auto saveResult = stbi_write_png(args.outputPath.value().c_str(), result.width, result.height, 1, converted.data(), result.width * sizeof(unsigned char));
        if (saveResult == 0) {
            throw std::runtime_error("Failed to save output image");
        }
    }
#else
    throw std::runtime_error("SVD support was not compiled");
#endif
}

void fsim(const IQM::Args& args) {
#ifdef COMPILE_FSIM
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    const IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::FSIM fsim(vulkan);

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = fsim.computeMetric(vulkan, input, reference);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    std::cout << "FSIM: " << result.fsim << std::endl << "FSIMc: " << result.fsimc << std::endl;

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FSIM support was not compiled");
#endif
}

void flip(const IQM::Args& args) {
#ifdef COMPILE_FLIP
    auto input = load_image(args.inputPath);
    auto reference = load_image(args.refPath);

    if (input.width != reference.width || input.height != reference.height) {
        throw std::runtime_error("Compared images must have the same size");
    }

    const IQM::GPU::VulkanRuntime vulkan;
    IQM::GPU::FLIP flip(vulkan);

    auto flip_args = IQM::GPU::FLIPArguments{};
    if (args.options.contains("FLIP_WIDTH")) {
        flip_args.monitor_width = std::stof(args.options.at("FLIP_WIDTH"));
    }
    if (args.options.contains("FLIP_RES")) {
        flip_args.monitor_resolution_x = std::stof(args.options.at("FLIP_RES"));
    }
    if (args.options.contains("FLIP_DISTANCE")) {
        flip_args.monitor_distance = std::stof(args.options.at("FLIP_DISTANCE"));
    }

    if (args.verbose) {
        std::cout << "Selected device: "<< vulkan.selectedDevice << std::endl
        << "FLIP monitor resolution: "<< flip_args.monitor_resolution_x << std::endl
        << "FLIP monitor distance: "<< flip_args.monitor_distance << std::endl
        << "FLIP monitor width: "<< flip_args.monitor_width << std::endl;
    }

    // starts only in debug, needs to init after vulkan
    initRenderDoc();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = flip.computeMetric(vulkan, input, reference, flip_args);
    auto end = std::chrono::high_resolution_clock::now();

    // saves capture for debugging
    finishRenderDoc();

    if (args.verbose) {
        result.timestamps.print(start, end);
    }
#else
    throw std::runtime_error("FLIP support was not compiled");
#endif
}

int main(int argc, const char **argv) {
    auto args = IQM::Args(argc, argv);
    if (args.verbose) {
        std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;
    }

    try {
        switch (args.method) {
            case IQM::Method::SSIM:
                ssim(args);
                break;
            case IQM::Method::CW_SSIM_CPU:
                break;
            case IQM::Method::SVD:
                svd(args);
                break;
            case IQM::Method::FSIM:
                fsim(args);
                break;
            case IQM::Method::FLIP:
                flip(args);
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }

    return 0;
}

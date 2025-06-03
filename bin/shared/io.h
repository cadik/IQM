/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_IO_H
#define IQM_IO_H

#include "stb_image.h"
#include "stb_image_write.h"

#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace IQM::Bin {
    struct Match {
        std::string testPath;
        std::string refPath;
        std::optional<std::string> outPath;
    };

    struct InputImage {
        unsigned width;
        unsigned height;
        std::vector<unsigned char> data;
    };

    inline InputImage load_image(const std::string &filename) {
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
            .width = static_cast<unsigned>(x),
            .height = static_cast<unsigned>(y),
            .data = std::move(dataVec)
        };
    }

    inline std::vector<unsigned char> convertFloatToChar(const std::vector<float>& data) {
        std::vector<unsigned char> result(data.size());

        for (unsigned long i = 0; i < data.size(); ++i) {
            result[i] = static_cast<unsigned char>(std::clamp(data[i], 0.0f, 1.0f) * 255.0f);
        }

        return result;
    }

    inline void save_char_image(const std::string &filename, const std::vector<unsigned char> &imageData, unsigned int width, unsigned int height) {
        auto saveResult = stbi_write_png(filename.c_str(), width, height, 1, imageData.data(), width * sizeof(unsigned char));
        if (saveResult == 0) {
            throw std::runtime_error("Failed to save output image");
        }
    }

    inline void save_float_image(const std::string &filename, const std::vector<float> &imageData, unsigned int width, unsigned int height) {
        const auto converted = convertFloatToChar(imageData);

        auto saveResult = stbi_write_png(filename.c_str(), width, height, 1, converted.data(), width * sizeof(unsigned char));
        if (saveResult == 0) {
            throw std::runtime_error("Failed to save output image");
        }
    }

    inline void save_color_image(const std::string &filename, const std::vector<unsigned char> &imageData, unsigned int width, unsigned int height) {
        auto saveResult = stbi_write_png(filename.c_str(), width, height, 4, imageData.data(), 4 * width * sizeof(unsigned char));
        if (saveResult == 0) {
            throw std::runtime_error("Failed to save output image");
        }
    }

    inline std::vector<float> load_model(const std::string &filename) {
        std::string path = std::filesystem::canonical("/proc/self/exe");
        path = path.substr(0, path.rfind('/') + 1);

        std::ifstream input(path + filename, std::ios::binary);

        if (input.bad()) {
            throw std::runtime_error("Failed to open file '" + filename + "'");
        }

        input.seekg(0, std::ios::end);
        const auto params = input.tellg() / sizeof(float);
        input.seekg(0, std::ios::beg);

        std::vector<float> buffer(params);
        input.read(reinterpret_cast<char*>(&buffer[0]), params*sizeof(float));

        return buffer;
    }
}

#endif //IQM_IO_H

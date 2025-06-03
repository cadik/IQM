/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "methods.h"
#include <stdexcept>

namespace IQM {
    std::string method_name(const Method &method) {
        switch (method) {
            case Method::PSNR:
                return "PSNR";
            case Method::SSIM:
                return "SSIM";
            case Method::CW_SSIM_CPU:
                return "CW-SSIM";
            case Method::SVD:
                return "SVD";
            case Method::FSIM:
                return "FSIM";
            case Method::FLIP:
                return "FLIP";
            case Method::LPIPS:
                return "LPIPS";
            default:
                throw std::runtime_error("unknown method");
        }
    }
}
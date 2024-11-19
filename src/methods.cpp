#include "methods.h"

namespace IQM {
    std::string method_name(const Method &method) {
        switch (method) {
            case Method::SSIM:
                return "SSIM";
            case Method::CW_SSIM_CPU:
                return "CW-SSIM";
            case Method::SVD:
                return "SVD";
            case Method::FSIM:
                return "FSIM";
            default:
                throw std::runtime_error("unknown method");
        }
    }
}
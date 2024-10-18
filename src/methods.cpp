#include "methods.h"

namespace IQM {
    std::string method_name(const Method &method) {
        if (method == Method::SSIM_CPU) {
            return "SSIM_CPU";
        }
        throw std::runtime_error("");
    }
}
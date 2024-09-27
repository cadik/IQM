#include "methods.h"

namespace IQM {
    std::string method_name(IQM::Method &method) {
        if (method == IQM::Method::SSIM_CPU) {
            return "SSIM_CPU";
        }
        throw std::runtime_error("");
    }
}
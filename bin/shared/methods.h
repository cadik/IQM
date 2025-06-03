/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_METHODS_H
#define IQM_METHODS_H

#include <string>

namespace IQM {
    enum class Method {
        PSNR = 0,
        SSIM = 1,
        CW_SSIM_CPU = 2,
        SVD = 3,
        FSIM = 4,
        FLIP = 5,
        LPIPS = 6,
    };

    std::string method_name(const Method &method);
};

#endif //IQM_METHODS_H

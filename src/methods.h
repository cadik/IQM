/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef IQM_METHODS_H
#define IQM_METHODS_H

#include <string>
#include <stdexcept>

namespace IQM {
    enum class Method {
        SSIM = 1,
        CW_SSIM_CPU = 2,
        SVD = 3,
        FSIM = 4,
    };

    std::string method_name(const Method &method);
};

#endif //IQM_METHODS_H

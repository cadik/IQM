#ifndef IQM_METHODS_H
#define IQM_METHODS_H

#include <string>
#include <stdexcept>

namespace IQM {
    enum class Method {
        SSIM_CPU = 0,
        SSIM = 1,
    };

    std::string method_name(const Method &method);
};

#endif //IQM_METHODS_H

#ifndef IQM_ARGS_H
#define IQM_ARGS_H

#include "methods.h"
#include <cstring>
#include <optional>
#include <stdexcept>

namespace IQM {
    class Args {
    public:
        Args(unsigned argc, const char* argv[]);
        Method method;
        std::string inputPath;
        std::string refPath;
        std::optional<std::string> outputPath;
    };
}

#endif //IQM_ARGS_H

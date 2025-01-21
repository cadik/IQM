/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_ARGS_H
#define IQM_ARGS_H

#include "methods.h"
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace IQM {
    class Args {
    public:
        Args(unsigned argc, const char* argv[]);
        Method method;
        std::string inputPath;
        std::string refPath;
        std::optional<std::string> outputPath;
        std::unordered_map<std::string, std::string> options;
        bool verbose = false;
    };
}

#endif //IQM_ARGS_H

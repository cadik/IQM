/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_PROFILEARGS_H
#define IQM_PROFILEARGS_H

#include "../shared/methods.h"
#include <optional>
#include <unordered_map>

namespace IQM {
    class ProfileArgs {
    public:
        ProfileArgs(unsigned argc, const char* argv[]);
        Method method;
        std::string inputPath;
        std::string refPath;
        std::unordered_map<std::string, std::string> options;
        std::optional<unsigned int> iterations;
        bool colorize = false;
        bool verbose = false;
        bool printHelp = false;
    };
}

#endif //IQM_PROFILEARGS_H

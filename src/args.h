#ifndef IQM_ARGS_H
#define IQM_ARGS_H

#include "methods.h"
#include <cstring>
#include <stdexcept>

namespace IQM {
    class Args {
    public:
        Args(unsigned argc, char* argv[]);
        Method method;
        std::string input_path;
        std::string ref_path;
    };
}

#endif //IQM_ARGS_H

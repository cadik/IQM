/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_BIN_FILE_MATCHER_H
#define IQM_BIN_FILE_MATCHER_H

#include "args.h"
#include "../shared/io.h"

#include <vector>

namespace IQM::Bin {
    class FileMatcher {
    public:
        std::vector<Match> match(const Args& args);
    };
}

#endif //IQM_BIN_FILE_MATCHER_H

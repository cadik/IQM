/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "file_matcher.h"

std::vector<IQM::Bin::Match> IQM::Bin::FileMatcher::match(const IQM::Bin::Args& args) {
    // add batch support after
    return {IQM::Bin::Match{.testPath = args.inputPath, .refPath = args.refPath, .outPath = args.outputPath}};
}

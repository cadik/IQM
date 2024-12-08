/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#include "args.h"
#include <cstring>

IQM::Args::Args(const unsigned argc, const char *argv[]) {
    bool parsedMethod = false;
    bool parsedInput = false;
    bool parsedReference = false;

    for (unsigned i = 0; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--method") == 0) {
                if (strcmp(argv[i + 1], "SSIM") == 0) {
                    this->method = Method::SSIM;
                    parsedMethod = true;
                } else if (strcmp(argv[i + 1], "CW_SSIM_CPU") == 0) {
                    this->method = Method::CW_SSIM_CPU;
                    parsedMethod = true;
                } else if (strcmp(argv[i + 1], "SVD") == 0) {
                    this->method = Method::SVD;
                    parsedMethod = true;
                } else if (strcmp(argv[i + 1], "FSIM") == 0) {
                    this->method = Method::FSIM;
                    parsedMethod = true;
                } else {
                    throw std::runtime_error("Unknown method");
                }
            } else if (strcmp(argv[i], "--input") == 0) {
                this->inputPath = std::string(argv[i + 1]);
                parsedInput = true;
            } else if (strcmp(argv[i], "--ref") == 0) {
                this->refPath = std::string(argv[i + 1]);
                parsedReference = true;
            } else if (strcmp(argv[i], "--output") == 0) {
                this->outputPath = std::string(argv[i + 1]);
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            this->verbose = true;
        }
    }

    if (!parsedMethod) {
        throw std::runtime_error("missing method");
    }
    if (!parsedInput) {
        throw std::runtime_error("missing input");
    }
    if (!parsedReference) {
        throw std::runtime_error("missing reference");
    }
}


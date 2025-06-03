/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include "args.h"
#include <cstring>
#include <stdexcept>

IQM::ProfileArgs::ProfileArgs(const unsigned argc, const char *argv[]) {
    bool parsedMethod = false;
    bool parsedInput = false;
    bool parsedReference = false;

    unsigned i = 1;
    while (i < argc)  {
        if (strcmp(argv[i], "--method") == 0) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing method argument");
            }
            if (strcmp(argv[i + 1], "SSIM") == 0) {
                this->method = Method::SSIM;
            } else if (strcmp(argv[i + 1], "CW_SSIM_CPU") == 0) {
                this->method = Method::CW_SSIM_CPU;
            } else if (strcmp(argv[i + 1], "SVD") == 0) {
                this->method = Method::SVD;
            } else if (strcmp(argv[i + 1], "FSIM") == 0) {
                this->method = Method::FSIM;
            } else if (strcmp(argv[i + 1], "FLIP") == 0) {
                this->method = Method::FLIP;
            } else if (strcmp(argv[i + 1], "PSNR") == 0) {
                this->method = Method::PSNR;
            } else if (strcmp(argv[i + 1], "LPIPS") == 0) {
                this->method = Method::LPIPS;
            } else {
                throw std::runtime_error("Unknown method");
            }
            parsedMethod = true;
            i+=2;
            continue;
        }

        if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing input argument");
            }

            this->inputPath = std::string(argv[i + 1]);
            parsedInput = true;
            i+=2;
            continue;
        }

        if (strcmp(argv[i], "--ref") == 0) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing reference argument");
            }

            this->refPath = std::string(argv[i + 1]);
            parsedReference = true;
            i+=2;
            continue;
        }

        if (strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "-i") == 0) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing iterations argument");
            }

            this->iterations = std::stoi(argv[i + 1]);
            i+=2;
            continue;
        }

        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            this->verbose = true;
            i += 1;
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            this->printHelp = true;
            return;
        }
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--colorize") == 0) {
            this->colorize = true;
            i += 1;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value argument");
        }
        this->options.emplace(std::string(argv[i]), std::string(argv[i + 1]));
        i+=2;
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

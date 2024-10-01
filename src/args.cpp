#include "args.h"

namespace IQM {
    IQM::Args::Args(unsigned argc, char *argv[]) {
        bool parsed_method = false;
        bool parsed_input = false;
        bool parsed_reference = false;

        for (unsigned i = 0; i < argc; i++) {
            if ((i + 1) < argc) {
                if (strcmp(argv[i], "--method") == 0) {
                    if (strcmp(argv[i + 1], "SSIM_CPU") == 0) {
                        this->method = Method::SSIM_CPU;
                        parsed_method = true;
                    }
                } else if (strcmp(argv[i], "--input") == 0) {
                    this->input_path = std::string(argv[i + 1]);
                    parsed_input = true;
                } else if (strcmp(argv[i], "--ref") == 0) {
                    this->input_path = std::string(argv[i + 1]);
                    parsed_reference = true;
                }
            }
        }

        if (!parsed_method) {
            throw std::runtime_error("missing method");
        }
        if (!parsed_input) {
            throw std::runtime_error("missing input");
        }
        if (!parsed_reference) {
            throw std::runtime_error("missing reference");
        }
    }
}


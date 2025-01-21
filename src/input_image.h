/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef INPUT_IMAGE_H
#define INPUT_IMAGE_H

#include <vector>

struct InputImage {
    int width;
    int height;
    std::vector<unsigned char> data;
};

#endif //INPUT_IMAGE_H

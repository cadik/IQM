/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#include "fsim_shared.glsl"

layout (local_size_x = 1, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer readonly InMedian {
    float inData[];
};

layout(std430, set = 0, binding = 1) buffer readonly InFilterSums {
    float inFilterSums[];
};

layout(std430, set = 0, binding = 2) buffer writeonly OutNoisePower {
    float outData[];
};

layout( push_constant ) uniform constants {
    uint size;
    uint index;
} push_consts;

void main() {
    uint x = gl_LocalInvocationID.x;

    float median;

    if (push_consts.size % 2 == 0) {
        median = (inData[push_consts.size / 2 - 1] + inData[push_consts.size / 2]) / 2;
    } else {
        median = inData[push_consts.size / 2];
    }

    float mean = -median / log(0.5);
    outData[push_consts.index] = mean / inFilterSums[push_consts.index % ORIENTATIONS];
}

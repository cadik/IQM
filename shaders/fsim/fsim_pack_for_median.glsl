/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#include "fsim_shared.glsl"

layout (local_size_x = 256, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer readonly InFFTBuf {
    float inData[];
};

layout(std430, set = 0, binding = 1) buffer writeonly OutFFTBuf {
    float outData[];
};

layout( push_constant ) uniform constants {
    uint size;
    uint index;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (x >= push_consts.size) {
        return;
    }

    uint base = 2 * push_consts.size * (OxS + push_consts.index * SCALES);

    float real = inData[base + 2 * x];
    float imag = inData[base + 2 * x + 1];

    float value = real * real + imag * imag;
    outData[x] = value;
}

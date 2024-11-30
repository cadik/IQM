/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define ORIENTATIONS 4
#define SCALES 4
#define OxS 16

layout (local_size_x = 32, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InFFTBuf {
    float inData[];
} fftbufs[OxS];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float sumFilter[ORIENTATIONS];
};

layout( push_constant ) uniform constants {
    // execution index
    uint index;
    uint size;
} push_consts;

shared float[32] subSums;

void main() {
    uint srcIndex = push_consts.index * ORIENTATIONS;
    uint maxSize = push_consts.size + 32;
    subSums[gl_LocalInvocationID.x] = 0;

    for (uint i = 0; i < maxSize; i+=32) {
        uint x = i + gl_LocalInvocationID.x;

        float val = mix(fftbufs[srcIndex].inData[x], 0, x >= push_consts.size);

        subSums[gl_LocalInvocationID.x] += pow(val, 2.0);
    }

    if (gl_LocalInvocationID.x == 0) {
        float totalSum = 0;
        for (uint i = 0; i < 32; i++) {
            totalSum += subSums[i];
        }
        sumFilter[push_consts.index] = totalSum;
    }
}
/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#define ORIENTATIONS 4
#define SCALES 4
#define OxS 16

layout (local_size_x = 256, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer readonly InFFTBuf {
    float inData[];
};

layout(std430, set = 0, binding = 1) buffer writeonly OutFFTBuf {
    float outData[];
};

layout( push_constant ) uniform constants {
    uint size;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (x >= push_consts.size) {
        return;
    }

    for (uint img = 0; img < 2; img++) {
        for (uint o = 0; o < ORIENTATIONS; o++) {
            uint index = img * ORIENTATIONS + o;
            uint base = 2 * push_consts.size * (OxS + index * SCALES);

            float real = inData[base + 2 * x];
            float imag = inData[base + 2 * x + 1];

            float value = real * real + imag * imag;
            outData[x + push_consts.size * index] = value;
        }
    }
}

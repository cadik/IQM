/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define ORIENTATIONS 4
#define SCALES 4
#define OxS 16

layout (local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer readonly InFFTBuf {
    float inData[];
};
layout(std430, set = 0, binding = 1) buffer writeonly Out {
    float outData[];
} outBuf[8];

layout( push_constant ) uniform constants {
    uint size;
} push_consts;

void main() {
    uint x = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) * 2;
    uint z = gl_WorkGroupID.z;

    if (x >= push_consts.size) {
        return;
    }

    // complex numbers
    uint subBufferSize = push_consts.size * 2;

    float sumAn2 = 0.0;
    for (uint i = 0; i < SCALES; i++) {
        uint inputIndex = subBufferSize * (i + z * ORIENTATIONS) + x;

        float value = pow(inData[inputIndex], 2.0) * push_consts.size;

        sumAn2 += value;
    }
    outBuf[z * 2].outData[x/2] = sumAn2;

    float sumAnCross = 0.0;
    for (uint i = 0; i < SCALES - 1; i++) {
        for (uint j = i + 1; j < SCALES; j++) {
            uint inputIndex1 = subBufferSize * (i + z * ORIENTATIONS) + x;
            uint inputIndex2 = subBufferSize * (j + z * ORIENTATIONS) + x;

            float value = inData[inputIndex1] * inData[inputIndex2] * push_consts.size;

            sumAnCross += value;
        }
    }
    outBuf[z * 2 + 1].outData[x/2] = sumAnCross;
}
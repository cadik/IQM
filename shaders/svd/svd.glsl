/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 128, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer inputBuf {
    float inData[];
};
layout(std430, set = 0, binding = 1) buffer outputBuf {
    float outData[];
};

layout( push_constant ) uniform constants {
    int size;
} push_consts;

shared float[128] subSums;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    subSums[gl_LocalInvocationID.x] = mix(inData[x], 0, x >= push_consts.size);

    memoryBarrierShared();
    barrier();

    // values are stored in groups of 8 side by side
    if ((gl_LocalInvocationID.x >> 3) % 2 == 0) {
        uint indexIn = gl_LocalInvocationID.x;
        uint offset = gl_LocalInvocationID.x % 8;
        uint indexOut = (((gl_LocalInvocationID.x >> 3) + 1) << 3) + offset;

        float srcVal = subSums[indexIn];
        float refVal = subSums[indexOut];

        subSums[gl_LocalInvocationID.x] = pow(srcVal - refVal, 2);
    }

    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x < 8) {
        float subSum = 0.0;
        for (x = 0; x < 8; x++) {
            subSum += subSums[x + gl_LocalInvocationID.x * 16];
        }
        outData[gl_WorkGroupID.x * 8 + gl_LocalInvocationID.x] = sqrt(subSum);
    }
}

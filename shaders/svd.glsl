#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 8, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer inputBuf {
    float inData[];
};
layout(std430, set = 0, binding = 1) buffer outputBuf {
    float outData[];
};

shared float[8] subSums;

void main() {
    uint x = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;

    float srcVal = inData[x];
    float refVal = inData[x + 8];

    subSums[gl_LocalInvocationID.x] = pow(srcVal - refVal, 2);

    if (gl_LocalInvocationID.x == 0) {
        float subSum = 0.0;
        for (x = 0; x < 8; x++) {
            subSum += subSums[x];
        }

        outData[gl_WorkGroupID.x] = sqrt(subSum);
    }
}
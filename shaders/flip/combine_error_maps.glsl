/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 1024, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
} inData[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
} outData;

layout( push_constant ) uniform constants {
    uint size;
} push_consts;

void main() {
    uint pixel = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (pixel >= push_consts.size) {
        return;
    }

    float deltaEf = inData[0].data[pixel];
    float deltaEc = inData[1].data[pixel];

    float value = pow(deltaEc, 1.0 - deltaEf);

    outData.data[pixel] = value;
}
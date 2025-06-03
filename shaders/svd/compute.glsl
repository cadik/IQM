/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D input_img[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
};

shared float[64] matLocal;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;

    uvec2 size = imageSize(input_img[z]);

    uint xLocal = gl_LocalInvocationID.x;
    uint yLocal = gl_LocalInvocationID.y;

    matLocal[xLocal + yLocal * 8] = imageLoad(input_img[z], ivec2(x, y)).x;

    memoryBarrierShared();
    barrier();

    // DO SVD HERE

    memoryBarrierShared();
    barrier();

    if (yLocal == 0) {
        uint index = x + y * size.x + (size.x * size.y) * z;

        data[index] = matLocal[xLocal];
    }
}
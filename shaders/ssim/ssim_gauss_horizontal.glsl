/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#include "ssim_shared.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform image2D input_img[5];
layout(set = 0, binding = 1, r32f) uniform image2D temp_img;

layout( push_constant ) uniform constants {
    int kernelSize;
    float sigma;
    int index;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[push_consts.index]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float total = 0.0;
    float totalWeight = 0.0;
    int start = -(push_consts.kernelSize - 1) / 2;
    int end = (push_consts.kernelSize - 1) / 2;

    // horizontal blur
    for (int xOffset = start; xOffset <= end; xOffset++) {
        int x = pos.x + xOffset;
        int y = pos.y;
        if (x >= size.x || x < 0) {
            continue;
        }
        float weight = gaussWeight(xOffset, push_consts.sigma);
        total += imageLoad(input_img[push_consts.index], ivec2(x, y)).x * weight;
        totalWeight += weight;
    }
    total /= totalWeight;

    imageStore(temp_img, pos, vec4(total, 0.0, 0.0, 0.0));
}

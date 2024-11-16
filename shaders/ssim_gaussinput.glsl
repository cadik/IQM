#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

#define E 2.71828182846
#define PI 3.141592653589

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rg32f) uniform readonly image2D input_img;
layout(set = 0, binding = 1, rg32f) uniform writeonly image2D output_img;

layout( push_constant ) uniform constants {
    int kernelSize;
    float sigma;
} push_consts;

float gaussWeight(ivec2 offset) {
    float dist = (offset.x * offset.x) + (offset.y * offset.y);
    return (1.0 / (2.0 * PI * pow(push_consts.sigma, 2.0))) * pow(E, -(dist / (2.0 * pow(push_consts.sigma, 2.0))));
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);

    if (x > imageSize(input_img).x || y > imageSize(input_img).y) {
        return;
    }

    ivec2 maxPos = imageSize(input_img);

    vec2 total = vec2(0.0);
    float totalWeight = 0.0;
    int start = -(push_consts.kernelSize - 1) / 2;
    int end = (push_consts.kernelSize - 1) / 2;
    for (int xOffset = start; xOffset <= end; xOffset++) {
        for (int yOffset = start; yOffset <= end; yOffset++) {
            int x = pos.x + xOffset;
            int y = pos.y + yOffset;
            if (x >= maxPos.x || y >= maxPos.y || x < 0 || y < 0) {
                continue;
            }
            float weight = gaussWeight(ivec2(xOffset, yOffset));
            total += imageLoad(input_img, ivec2(x, y)).xy * weight;
            totalWeight += weight;
        }
    }

    total /= totalWeight;

    imageStore(output_img, pos, vec4(total.x, total.y, 0.0, 0.0));
}
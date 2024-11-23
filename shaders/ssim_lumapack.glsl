/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

#define E 2.71828182846
#define PI 3.141592653589

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D ref_img;
layout(set = 0, binding = 2, rg32f) uniform writeonly image2D output_img;

// Rec. 601 - same as openCV
float luminance(vec4 color) {
    return 0.299 * color.r + 0.581 * color.g + 0.114 * color.b;
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);

    if (x >= imageSize(input_img).x || y >= imageSize(input_img).y) {
        return;
    }

    float lumaSrc = luminance(imageLoad(input_img, pos));
    float lumaRef = luminance(imageLoad(ref_img, pos));

    imageStore(output_img, pos, vec4(lumaSrc, lumaRef, 0.0, 0.0));
}
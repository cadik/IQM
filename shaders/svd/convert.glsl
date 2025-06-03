/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, r32f) uniform writeonly image2D output_img[2];

// Rec. 601 - same as openCV
float luminance(vec4 color) {
    return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);

    if (x >= imageSize(input_img[0]).x || y >= imageSize(input_img[0]).y) {
        return;
    }

    float lumaSrc = luminance(imageLoad(input_img[0], pos));
    float lumaRef = luminance(imageLoad(input_img[1], pos));

    imageStore(output_img[0], pos, vec4(lumaSrc, 0.0, 0.0, 0.0));
    imageStore(output_img[1], pos, vec4(lumaRef, 0.0, 0.0, 0.0));
}
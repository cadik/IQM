/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D color_map;
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D output_img;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[0]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float deltaEf = imageLoad(input_img[0], pos).x;
    float deltaEc = imageLoad(input_img[1], pos).x;

    float value = pow(deltaEc, 1.0 - deltaEf);

    vec4 color = imageLoad(color_map, ivec2(int(floor(value * 255.0)), 0));

    imageStore(output_img, pos, color);
}
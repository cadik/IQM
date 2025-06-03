/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D input_img;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D output_img;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D color_map;

layout( push_constant ) uniform constants {
    bool invert;
    float scaler;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float value = clamp(imageLoad(input_img, pos).x, 0.0, 1.0);

    if (push_consts.invert) {
        value = 1.0 - value;
    }

    value = pow(value, 1.0 / push_consts.scaler);
    int colorMax = imageSize(color_map).x - 1;
    vec4 color = imageLoad(color_map, ivec2(int(floor(value * colorMax)), 0));

    imageStore(output_img, pos, color);
}

/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D lowpass_filter;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D out_filter;

layout( push_constant ) uniform constants {
    int scale;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(lowpass_filter);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float gradX = (float(x) / float(size.x)) * push_consts.scale;

    imageStore(out_filter, pos, vec4(gradX));
}
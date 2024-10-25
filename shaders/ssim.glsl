#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform image2D input_img;
layout(set = 0, binding = 1, rgba8) uniform image2D ref_img;
layout(set = 0, binding = 2, rgba8) uniform image2D output_img;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    imageStore(output_img, ivec2(x, y), vec4(gl_LocalInvocationID.x / 16.0, gl_LocalInvocationID.y / 16.0, 0.0, 1.0));
}
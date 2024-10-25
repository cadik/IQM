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
    ivec2 pos = ivec2(x, y);

    imageStore(output_img, pos, vec4(abs(imageLoad(input_img, pos).xyz - imageLoad(ref_img, pos).xyz), 1.0));
}
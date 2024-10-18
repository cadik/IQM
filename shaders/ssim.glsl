#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform image2D input_img;
layout(set = 0, binding = 1, rgba8) uniform image2D ref_img;
layout(set = 0, binding = 2, rgba8) uniform image2D output_img;

void main() {
    int k = 0;
    for(int i = 0; i < 1024000; i++) {
        k += i;
    }
}
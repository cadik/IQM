#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 8, local_size_y = 8) in;

// each pixel is [I, Q, Y, 1], where I, Q, Y are FSIM color values
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img;
layout(std430, set = 0, binding = 1) buffer outputBuf {
    float outData[];
};

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 size = imageSize(input_img);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    outData[x + size.x * y] = imageLoad(input_img, pos).z;
}
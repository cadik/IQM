#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D filter_img;

layout( push_constant ) uniform constants {
    // should be in 0 - 0.5
    float cutoff;
    // n >= 1
    int n;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_img);

    if (x >= size.x || y >= size.y) {
        return;
    }

    // do the equivalent of ifftshift first, so writes are ordered
    uint shiftedX = (x + uint(size.x)/2) % size.x;
    uint shiftedY = (y + uint(size.y)/2) % size.y;
    float scaledX = float(int(shiftedX) - size.x/2) / float(size.x);
    float scaledY = float(int(shiftedY) - size.y/2) / float(size.y);

    float radius = sqrt((scaledX * scaledX) + (scaledY * scaledY));

    float res = 1.0 / (1.0 + pow(radius / push_consts.cutoff, 2.0 * float(push_consts.n)));

    imageStore(filter_img, pos, vec4(res, 0.0, 0.0, 0.0));
}
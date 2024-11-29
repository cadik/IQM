/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define ORIENTATIONS 4
#define SCALES 4
#define OxS 16

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D gabor_filters[SCALES];
layout(set = 0, binding = 1, r32f) uniform readonly image2D angular_filters[ORIENTATIONS];
layout(std430, set = 0, binding = 2) buffer OutFFTBuf {
    float outData[];
} fftbufs[OxS];

layout( push_constant ) uniform constants {
    // execution index
    uint index;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    uint gabor_index = push_consts.index % SCALES;
    uint angular_index = push_consts.index / ORIENTATIONS;

    ivec2 size = imageSize(gabor_filters[gabor_index]);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float gabor = imageLoad(gabor_filters[gabor_index], pos).x;
    float angular = imageLoad(angular_filters[angular_index], pos).x;

    fftbufs[push_consts.index].outData[(x + size.x * y) * 2] = gabor * angular;
    fftbufs[push_consts.index].outData[(x + size.x * y) * 2 + 1] = 0.0;
}
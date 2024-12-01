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

layout(set = 0, binding = 0, r32f) uniform readonly image2D angular_filters[SCALES];
layout(set = 0, binding = 1, r32f) uniform readonly image2D gabor_filters[ORIENTATIONS];
layout(std430, set = 0, binding = 2) buffer readonly InFFTBuf {
    float inData[];
};
layout(std430, set = 0, binding = 3) buffer writeonly OutFFTBuf {
    float outData[];
};

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

    uint offset = push_consts.index * size.x * size.y * 2;
    uint stride = size.x * size.y * 2 * OxS;

    if (x >= size.x || y >= size.y) {
        return;
    }

    uint pixelIndex = (x + size.x * y) * 2;

    float gabor = imageLoad(gabor_filters[gabor_index], pos).x;
    float angular = imageLoad(angular_filters[angular_index], pos).x;
    float src = inData[pixelIndex];
    float srcImg = inData[pixelIndex + 1];
    float ref = inData[pixelIndex + (size.x * size.y * 2)];
    float refImg = inData[pixelIndex + (size.x * size.y * 2) + 1];

    outData[pixelIndex + offset] = gabor * angular;
    outData[pixelIndex + 1 + offset] = 0.0;
    outData[pixelIndex + offset + stride] = gabor * angular * src;
    outData[pixelIndex + 1 + offset + stride] = gabor * angular * srcImg;
    outData[pixelIndex + offset + stride * 2] = gabor * angular * ref;
    outData[pixelIndex + 1 + offset + stride * 2] = gabor * angular * refImg;
}
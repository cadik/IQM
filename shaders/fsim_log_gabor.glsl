/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define SCALES 4

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D lowpass_filter;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D out_filter[SCALES];

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(lowpass_filter);

    if (x >= size.x || y >= size.y) {
        return;
    }

    int minWaveLength = 6;
    int scaler = 2;
    float sigmaOnf = 0.55;

    float wavelength = minWaveLength * pow(scaler, z);

    // do the equivalent of ifftshift first, so writes are ordered
    uint shiftedX = (x + uint(size.x)/2) % size.x;
    uint shiftedY = (y + uint(size.y)/2) % size.y;
    float scaledX = float(int(shiftedX) - size.x/2) / float(size.x);
    float scaledY = float(int(shiftedY) - size.y/2) / float(size.y);

    float radius = sqrt((scaledX * scaledX) + (scaledY * scaledY));
    float frequency = 1.0 / wavelength;
    float value = exp(-pow(log(radius/frequency), 2.0) / (2.0 * pow(log(sigmaOnf), 2.0)));

    float lowpass = imageLoad(lowpass_filter, pos).x;
    imageStore(out_filter[z], pos, vec4(value * lowpass));

    if(x == 0 && y == 0) {
        imageStore(out_filter[z], pos, vec4(0));
    }
}
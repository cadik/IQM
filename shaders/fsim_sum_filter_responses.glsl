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

// each pixel is [Am, Energy, 0, 0]
layout(set = 0, binding = 0, rg32f) uniform writeonly image2D filter_responses_input[ORIENTATIONS];
// each pixel is [Am, Energy, 0, 0]
layout(set = 0, binding = 1, rg32f) uniform writeonly image2D filter_responses_ref[ORIENTATIONS];
layout(std430, set = 0, binding = 2) buffer readonly InFFTBuf {
    float inData[];
};

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_responses_input[z]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    vec4 sumsIn = vec4(0.0);
    vec4 sumsRef = vec4(0.0);

    uint floatsPerImage = size.x * size.y * 2;
    uint stride = floatsPerImage * OxS;

    uint pixelOffset = (pos.x + pos.y * size.x) * 2;
    uint orientationOffset = floatsPerImage * z * ORIENTATIONS;

    for (uint i = 0; i < SCALES; i++) {
        uint scaleOffset = i * floatsPerImage;

        float realSrc = inData[pixelOffset + scaleOffset + orientationOffset + stride];
        float imSrc = inData[pixelOffset + scaleOffset + orientationOffset + stride + 1];
        sumsIn.x += sqrt(pow(realSrc, 2.0) + pow(imSrc, 2.0));
        sumsIn.y += realSrc;
        sumsIn.z += imSrc;

        float realRef = inData[pixelOffset + scaleOffset + orientationOffset + stride * 2];
        float imRef = inData[pixelOffset + scaleOffset + orientationOffset + stride * 2 + 1];
        sumsRef.x += sqrt(pow(realRef, 2.0) + pow(imRef, 2.0));
        sumsRef.y += realRef;
        sumsRef.z += imRef;
    }

    float XEnergyIn = sqrt(pow(sumsIn.y, 2.0) + pow(sumsIn.z, 2.0)) + 0.0001;
    sumsIn.y /= XEnergyIn;
    sumsIn.z /= XEnergyIn;

    float XEnergyRef = sqrt(pow(sumsRef.y, 2.0) + pow(sumsRef.z, 2.0)) + 0.0001;
    sumsRef.y /= XEnergyRef;
    sumsRef.z /= XEnergyRef;

    float energyIn = 0.0;
    float energyRef = 0.0;

    for (uint i = 0; i < SCALES; i++) {
        uint scaleOffset = i * floatsPerImage;

        float realSrc = inData[pixelOffset + scaleOffset + orientationOffset + stride];
        float imSrc = inData[pixelOffset + scaleOffset + orientationOffset + stride + 1];
        float realRef = inData[pixelOffset + scaleOffset + orientationOffset + stride * 2];
        float imRef = inData[pixelOffset + scaleOffset + orientationOffset + stride * 2 + 1];

        energyIn += realSrc * sumsIn.y + imSrc * sumsIn.z - abs(realSrc * sumsIn.z - imSrc * sumsIn.y);
        energyRef += realRef * sumsRef.y + imRef * sumsRef.z - abs(realRef * sumsRef.z - imRef * sumsRef.y);
    }

    imageStore(filter_responses_input[z], pos, vec4(sumsIn.x, energyIn, 0.0, 0.0));
    imageStore(filter_responses_ref[z], pos, vec4(sumsRef.x, energyRef, 0.0, 0.0));
}
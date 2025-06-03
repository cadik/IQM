/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 1024, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
} inData[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
} outData[2];

layout(set = 0, binding = 2, rgba32f) uniform readonly image2D filter_img;

layout( push_constant ) uniform constants {
    uint size;
    uint width;
    uint height;
} push_consts;

void main() {
    uint pixel = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (pixel >= push_consts.size) {
        return;
    }

    uint x = pixel % push_consts.width;
    uint y = pixel / push_consts.width;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 filterSize = imageSize(filter_img);

    float qf = 0.5;

    vec2 edgeInput = vec2(0.0);
    vec2 edgeRef = vec2(0.0);
    vec2 pointInput = vec2(0.0);
    vec2 pointRef = vec2(0.0);

    float dx = 0.0;
    float ddx = 0.0;
    float value = 0.0;

    for (int j = -filterSize.x / 2; j <= filterSize.x/2; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, int(push_consts.width) - 1));
        uint filterX = j + filterSize.x / 2;

        vec3 filterWeights = imageLoad(filter_img, ivec2(filterX, 0)).xyz;

        uint index = (actualX + y * push_consts.width) * 3;
        float inValue = (inData[z].data[index] + 16.0) / 116.0;

        value += filterWeights.x * inValue;
        dx += filterWeights.y * inValue;
        ddx += filterWeights.z * inValue;
    }

    outData[z].data[pixel * 3] = dx;
    outData[z].data[pixel * 3 + 1] = ddx;
    outData[z].data[pixel * 3 + 2] = value;
}
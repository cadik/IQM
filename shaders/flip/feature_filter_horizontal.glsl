/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_img[2];
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D filter_img;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[0]);
    ivec2 filterSize = imageSize(filter_img);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float qf = 0.5;

    vec2 edgeInput = vec2(0.0);
    vec2 edgeRef = vec2(0.0);
    vec2 pointInput = vec2(0.0);
    vec2 pointRef = vec2(0.0);

    float dx = 0.0;
    float ddx = 0.0;
    float value = 0.0;

    for (int j = -filterSize.x / 2; j <= filterSize.x/2; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, size.x - 1));
        uint filterX = j + filterSize.x / 2;

        vec3 filterWeights = imageLoad(filter_img, ivec2(filterX, 0)).xyz;
        float inValue = (imageLoad(input_img[z], ivec2(actualX, y)).x + 16.0) / 116.0;

        value += filterWeights.x * inValue;
        dx += filterWeights.y * inValue;
        ddx += filterWeights.z * inValue;
    }

    imageStore(output_img[z], pos, vec4(dx, ddx, value, 1.0));
}
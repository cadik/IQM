/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#include "ssim_shared.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D input_img[5];
layout(set = 0, binding = 1, r32f) uniform writeonly image2D output_img;

layout( push_constant ) uniform constants {
    int kernelSize;
    float k_1;
    float k_2;
    float sigma;
} push_consts;

void main() {
    float c_1 = pow(push_consts.k_1, 2);
    float c_2 = pow(push_consts.k_2, 2);

    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 maxPos = imageSize(input_img[0]);
    ivec2 pos = ivec2(x, y);

    if (x >= maxPos.x || y >= maxPos.y) {
        return;
    }

    float meanImg = imageLoad(input_img[0], pos).x;
    float meanRef = imageLoad(input_img[1], pos).x;

    float varInput = imageLoad(input_img[2], pos).x - (meanImg * meanImg);
    float varRef = imageLoad(input_img[3], pos).x - (meanRef * meanRef);
    float coVar = imageLoad(input_img[4], pos).x - (meanImg * meanRef);

    float smallPart = (2.0 * coVar + c_2) /  (varInput + varRef + c_2);
    float bigPart = (2.0 * meanImg * meanRef + c_1) / (pow(meanImg, 2.0) + pow(meanRef, 2.0) + c_1);
    float outCol = smallPart * bigPart;

    imageStore(output_img, pos, vec4(vec3(outCol), 1.0));
}
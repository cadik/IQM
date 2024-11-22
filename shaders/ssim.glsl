#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

#define E 2.71828182846
#define PI 3.141592653589

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rg32f) uniform readonly image2D luma_img;
layout(set = 0, binding = 1, rg32f) uniform readonly image2D lumaBlur_img;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D output_img;

layout( push_constant ) uniform constants {
    int kernelSize;
    float k_1;
    float k_2;
    float sigma;
} push_consts;

float gaussWeight(ivec2 offset) {
    float dist = (offset.x * offset.x) + (offset.y * offset.y);
    return pow(E, -(dist / (2.0 * pow(push_consts.sigma, 2.0))));
}

#define gaussFunc(e) ivec2 maxPos = imageSize(luma_img); \
    float total = 0.0; \
    float totalWeight = 0.0; \
    int start = -(push_consts.kernelSize - 1) / 2; \
    int end = (push_consts.kernelSize - 1) / 2; \
    for (int xOffset = start; xOffset <= end; xOffset++) { \
        for (int yOffset = start; yOffset <= end; yOffset++) { \
            int x = pos.x + xOffset; \
            int y = pos.y + yOffset; \
            if (x >= maxPos.x || y >= maxPos.y || x < 0 || y < 0) { \
                continue; \
            } \
            float weight = gaussWeight(ivec2(xOffset, yOffset)); \
            total += e * weight; \
            totalWeight += weight; \
        } \
    } \
    return total / totalWeight;

float gaussInputVar(ivec2 pos) {
    gaussFunc( pow(imageLoad(luma_img, ivec2(x, y)).x - imageLoad(lumaBlur_img, pos).x , 2.0) )
}

float gaussRefVar(ivec2 pos) {
    gaussFunc( pow(imageLoad(luma_img, ivec2(x, y)).y - imageLoad(lumaBlur_img, pos).y , 2.0) )
}

float gaussCoVar(ivec2 pos) {
    gaussFunc(( imageLoad(luma_img, ivec2(x, y)).x - imageLoad(lumaBlur_img, pos).x) * (imageLoad(luma_img, ivec2(x, y)).y - imageLoad(lumaBlur_img, pos).y) )
}

void main() {
    float c_1 = pow(push_consts.k_1, 2);
    float c_2 = pow(push_consts.k_2, 2);

    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);

    if (x >= imageSize(luma_img).x || y >= imageSize(luma_img).y) {
        return;
    }

    float meanImg = imageLoad(lumaBlur_img, pos).x;
    float meanRef = imageLoad(lumaBlur_img, pos).y;
    float varInput = gaussInputVar(pos);
    float varRef = gaussRefVar(pos);
    float coVar = gaussCoVar(pos);

    float outCol = ((2.0 * meanImg * meanRef + c_1) * (2.0 * coVar + c_2)) /
        ((pow(meanImg, 2.0) + pow(meanRef, 2.0) + c_1) * (varInput + varRef + c_2));

    imageStore(output_img, pos, vec4(vec3(outCol), 1.0));
}
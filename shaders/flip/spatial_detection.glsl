/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, r32f) uniform writeonly image2D output_img;

const mat3 RGB_TO_XYZ = mat3(
    float(10135552) / 24577794, float(8788810) / 24577794, float(4435075) / 24577794,
    float(2613072) / 12288897, float(8788810) / 12288897, float(887015) / 12288897,
    float(1425312) / 73733382, float(8788810) / 73733382, float(70074185) / 73733382
);

// works in L*a*b*
float hyABDistance(vec3 a, vec3 b) {
    vec3 delta = a - b;
    return abs(delta.r) + sqrt(pow(delta.g, 2.0) + pow(delta.b, 2.0));
}

vec3 linearRgbToLab(vec3 color) {
    vec3 xyz = color * RGB_TO_XYZ;
    vec3 ref = vec3(1.0) * RGB_TO_XYZ;

    xyz /= ref;

    // convert to L*a*b
    float delta = 6.0 / 29.0;
    float limit = 0.008856;

    vec3 aboveLimit = vec3(float(xyz.x > limit), float(xyz.y > limit), float(xyz.z > limit));
    vec3 above = vec3(pow(xyz.x, 1.0 / 3.0), pow(xyz.y, 1.0 / 3.0), pow(xyz.z, 1.0 / 3.0));
    vec3 below = xyz / (3.0 * delta * delta) + 4.0 / 29.0;

    vec3 colorOut = mix(below, above, aboveLimit);

    return vec3(116.0 * colorOut.y - 16.0, 500.0 * (colorOut.x - colorOut.y), 200.0 * (colorOut.y - colorOut.z));
}

float remapError(float delta, float cmax) {
    float pc = 0.4;
    float pt = 0.95;

    float pccmax = pc * cmax;

    float above = pt + ((delta - pccmax) / (cmax - pccmax)) * (1.0 - pt);
    float below = (pt / pccmax) * delta;

    return mix(above, below, delta < pccmax);
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[0]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float qc = 0.7;

    vec3 inp = imageLoad(input_img[0], pos).xyz;
    vec3 ref = imageLoad(input_img[1], pos).xyz;

    float deltaEhyab = hyABDistance(inp, ref);
    vec3 huntAdjGreen = linearRgbToLab(vec3(0.0, 1.0, 0.0));
    huntAdjGreen.y *= 0.01 * huntAdjGreen.x;
    huntAdjGreen.z *= 0.01 * huntAdjGreen.x;

    vec3 huntAdjBlue = linearRgbToLab(vec3(0.0, 0.0, 1.0));
    huntAdjBlue.y *= 0.01 * huntAdjBlue.x;
    huntAdjBlue.z *= 0.01 * huntAdjBlue.x;

    float cmax = pow(hyABDistance(huntAdjGreen, huntAdjBlue), qc);
    float deltaEc = remapError(pow(deltaEhyab, qc), cmax);

    imageStore(output_img, pos, vec4(deltaEc));
}
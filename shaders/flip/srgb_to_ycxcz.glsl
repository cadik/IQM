/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_img[2];

#define SRGB_LIMIT 0.04045

const mat3 RGB_TO_XYZ = mat3(
    float(10135552) / 24577794, float(8788810) / 24577794, float(4435075) / 24577794,
    float(2613072) / 12288897, float(8788810) / 12288897, float(887015) / 12288897,
    float(1425312) / 73733382, float(8788810) / 73733382, float(70074185) / 73733382
);

vec3 srgb_to_linear_rgb(vec3 color) {
    vec3 aboveLimit = vec3(float(color.r > SRGB_LIMIT), float(color.g > SRGB_LIMIT), float(color.b > SRGB_LIMIT));
    vec3 outColorAbove = color / 12.92;
    vec3 outColorBelow = pow((color + 0.055) / 1.055, vec3(2.4));

    return mix(outColorAbove, outColorBelow, aboveLimit);
}

vec3 xyz_to_ycxcz(vec3 color) {
    vec3 ref = RGB_TO_XYZ * vec3(1.0);

    color /= ref;

    float y = 116 * color.g - 16;
    float cx = 500 * (color.r - color.g);
    float cz = 200 * (color.g - color.b);

    return vec3(y, cx, cz);
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);

    if (x >= imageSize(input_img[z]).x || y >= imageSize(input_img[z]).y) {
        return;
    }

    vec3 inputColor = imageLoad(input_img[z], pos).xyz;

    vec3 transformedColor = xyz_to_ycxcz(RGB_TO_XYZ * srgb_to_linear_rgb(inputColor));

    imageStore(output_img[z], pos, vec4(transformedColor, 1.0));
}
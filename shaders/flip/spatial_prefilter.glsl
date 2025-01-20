/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D filter_img;
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D output_img[2];

const mat3 XYZ_TO_RGB = mat3(
    3.241003275, -1.537398934, -0.498615861,
    -0.969224334, 1.875930071, 0.041554224,
    0.055639423, -0.204011202, 1.057148933
);

const mat3 RGB_TO_XYZ = mat3(
    float(10135552) / 24577794, float(8788810) / 24577794, float(4435075) / 24577794,
    float(2613072) / 12288897, float(8788810) / 12288897, float(887015) / 12288897,
    float(1425312) / 73733382, float(8788810) / 73733382, float(70074185) / 73733382
);

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[z]);

    if (x >= imageSize(input_img[z]).x || y >= imageSize(input_img[z]).y) {
        return;
    }

    int halfSize = imageSize(filter_img).x / 2;

    vec3 opponent = vec3(0.0);

    for (int j = -halfSize; j <= halfSize; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, size.x - 1));
        uint filterX = j + halfSize;

        for (int k = -halfSize; k <= halfSize; k++) {
            uint actualY = uint(clamp(int(y) - k, 0, size.y - 1));
            uint filterY = k + halfSize;

            vec3 ycc = imageLoad(input_img[z], ivec2(actualX, actualY)).xyz;

            vec3 filter_val = imageLoad(filter_img, ivec2(filterX, filterY)).xyz;

            opponent += ycc * filter_val;
        }
    }

    float yy = (opponent.x + 16.0) / 116.0;
    float cx = (opponent.y) / 500.0;
    float cz = (opponent.z) / 200.0;

    vec3 ref = vec3(1.0) * RGB_TO_XYZ;
    vec3 xyz = vec3(yy + cx, yy, yy - cz) * ref;

    vec3 linRgb = clamp(xyz * XYZ_TO_RGB, vec3(0.0), vec3(1.0));

    xyz = linRgb * RGB_TO_XYZ;
    xyz /= ref;

    // convert to L*a*b
    float delta = 6.0 / 29.0;
    float limit = 0.008856;

    vec3 aboveLimit = vec3(float(xyz.x > limit), float(xyz.y > limit), float(xyz.z > limit));
    vec3 above = vec3(pow(xyz.x, 1.0 / 3.0), pow(xyz.y, 1.0 / 3.0), pow(xyz.z, 1.0 / 3.0));
    vec3 below = xyz / (3.0 * delta * delta) + 4.0 / 29.0;

    vec3 color = mix(below, above, aboveLimit);

    vec3 lab = vec3(116.0 * color.y - 16.0, 500.0 * (color.x - color.y), 200.0 * (color.y - color.z));
    // Hunt effect adjustment
    lab.y *= 0.01 * lab.x;
    lab.z *= 0.01 * lab.x;

    imageStore(output_img[z], pos, vec4(lab, 0.0));
}
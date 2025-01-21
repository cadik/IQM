/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#define PI 3.141592653589

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_img[2];

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

const vec4 lumaParams = vec4(1.0, 0.0047, 0, 0.00001);
const vec4 rgParams = vec4(1.0, 0.0053, 0, 0.00001);
const vec4 byParams = vec4(34.1, 0.04, 13.5, 0.025);

float getGaussValue(float d, vec4 par) {
    return par.x * sqrt(PI / par.y) * exp(-pow(PI, 2.0) * d / par.y) + par.z * sqrt(PI / par.w) * exp(-pow(PI, 2.0) * d / par.w);
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[z]);

    if (x >= imageSize(input_img[z]).x || y >= imageSize(input_img[z]).y) {
        return;
    }

    int radius = int(ceil(3.0 * sqrt(0.04 / (2.0 * PI * PI)) * push_consts.pixels_per_degree));
    int halfSize = radius;
    float deltaX = 1.0 / push_consts.pixels_per_degree;

    vec3 opponent = vec3(0.0);
    vec3 opponentTotal = vec3(0.0);

    for (int j = -halfSize; j <= halfSize; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, size.x - 1));
        int k = 0;

        vec3 ycc = imageLoad(input_img[z], ivec2(actualX, y)).xyz;

        float xx = float(j) * deltaX;
        float d = xx * xx;

        vec3 filter_val = vec3(getGaussValue(d, lumaParams), getGaussValue(d, rgParams), getGaussValue(d, byParams));

        opponent += ycc * filter_val;
        opponentTotal += filter_val;
    }

    opponent /= opponentTotal;

    imageStore(output_img[z], pos, vec4(opponent, 0.0));
}
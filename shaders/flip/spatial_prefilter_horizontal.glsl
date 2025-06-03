/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#define PI 3.141592653589

layout (local_size_x = 1024, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
} inData[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
} outData;

layout( push_constant ) uniform constants {
    float pixels_per_degree;
    uint index;
    uint size;
    uint width;
    uint height;
} push_consts;

const vec4 lumaParams = vec4(1.0, 0.0047, 0, 0.00001);
const vec4 rgParams = vec4(1.0, 0.0053, 0, 0.00001);
const vec4 byParams = vec4(34.1, 0.04, 13.5, 0.025);

float getGaussValue(float d, vec4 par) {
    return par.x * sqrt(PI / par.y) * exp(-pow(PI, 2.0) * d / par.y) + par.z * sqrt(PI / par.w) * exp(-pow(PI, 2.0) * d / par.w);
}

void main() {
    uint pixel = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (pixel >= push_consts.size) {
        return;
    }

    uint x = pixel % push_consts.width;
    uint y = pixel / push_consts.width;

    uint z = push_consts.index;
    ivec2 pos = ivec2(x, y);

    int radius = int(ceil(3.0 * sqrt(0.04 / (2.0 * PI * PI)) * push_consts.pixels_per_degree));
    int halfSize = radius;
    float deltaX = 1.0 / push_consts.pixels_per_degree;

    vec3 opponent = vec3(0.0);
    vec3 opponentTotal = vec3(0.0);

    for (int j = -halfSize; j <= halfSize; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, int(push_consts.width) - 1));
        int k = 0;

        uint index = (actualX + y * push_consts.width) * 3;
        vec3 ycc = vec3(inData[z].data[index], inData[z].data[index + 1], inData[z].data[index + 2]);

        float xx = float(j) * deltaX;
        float d = xx * xx;

        vec3 filter_val = vec3(getGaussValue(d, lumaParams), getGaussValue(d, rgParams), getGaussValue(d, byParams));

        opponent += ycc * filter_val;
        opponentTotal += filter_val;
    }

    opponent /= opponentTotal;

    outData.data[pixel * 3] = opponent.x;
    outData.data[pixel * 3 + 1] = opponent.y;
    outData.data[pixel * 3 + 2] = opponent.z;
}
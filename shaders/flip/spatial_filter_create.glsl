/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#define PI 3.141592653589

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D filter_img[3];

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

vec4 params() {
    if (gl_WorkGroupID.z == 0) {
        return vec4(1.0, 0.0047, 0, 0.00001);
    } else if (gl_WorkGroupID.z == 1) {
        return vec4(1.0, 0.0053, 0, 0.00001);
    } else {
        return vec4(34.1, 0.04, 13.5, 0.025);
    }
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_img[z]);

    if (x >= imageSize(filter_img[z]).x || y >= imageSize(filter_img[z]).y) {
        return;
    }

    int radius = int(ceil(3.0 * sqrt(0.04 / (2.0 * PI * PI)) * push_consts.pixels_per_degree));

    int xCoord = int(x) - radius;
    int yCoord = int(y) - radius;
    float deltaX = 1.0 / push_consts.pixels_per_degree;

    float xx = float(xCoord) * deltaX;
    float yy = float(yCoord) * deltaX;
    float d = xx * xx + yy * yy;

    vec4 par = params();

    float value = par.x * sqrt(PI / par.y) * exp(-pow(PI, 2.0) * d / par.y) + par.z * sqrt(PI / par.w) * exp(-pow(PI, 2.0) * d / par.w);

    imageStore(filter_img[z], pos, vec4(value));
}
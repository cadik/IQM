/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D filter_img[2];

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_img[z]);

    if (x >= imageSize(filter_img[z]).x || y >= imageSize(filter_img[z]).y) {
        return;
    }

    float w = 0.082;
    float sd = 0.5 * w * push_consts.pixels_per_degree;
    int radius = int(ceil(3.0 * sd));

    int xCoord = int(x) - radius;
    int yCoord = int(y) - radius;

    float g = exp( -(pow(float(xCoord), 2.0) + pow(float(yCoord), 2.0)) / (2.0 * pow(sd, 2.0)));

    float value;
    if (z == 0) {
        // edges
        value = -xCoord * g;
    } else {
        // points
        value = (pow(float(xCoord), 2.0) / pow(sd, 2.0) - 1) * g;
    }

    imageStore(filter_img[z], pos, vec4(value));
}
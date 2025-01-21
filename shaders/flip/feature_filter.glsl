/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 1) in;

layout(set = 0, binding = 0, rgba32f) uniform writeonly image2D filter_img;

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    ivec2 pos = ivec2(x, 0);
    ivec2 size = imageSize(filter_img);

    if (x >= size.x) {
        return;
    }

    float w = 0.082;
    float sd = 0.5 * w * push_consts.pixels_per_degree;
    int radius = int(ceil(3.0 * sd));

    int xCoord = int(x) - radius;

    float g = exp( -(pow(float(xCoord), 2.0)) / (2.0 * pow(sd, 2.0)));

    float edge = -xCoord * g;
    float point = (pow(float(xCoord), 2.0) / pow(sd, 2.0) - 1) * g;

    imageStore(filter_img, pos, vec4(g, edge, point, 1.0));
}
/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform image2D filter_img;

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_img);

    if (x >= imageSize(filter_img).x || y >= imageSize(filter_img).y) {
        return;
    }

    vec3 sum = vec3(0.0);

    for (int j = 0; j < size.x; j++) {
        for (int k = 0; k < size.y; k++) {
            vec3 value = imageLoad(filter_img, ivec2(j, k)).xyz;
            sum += value;
        }
    }

    vec3 pixelValue = imageLoad(filter_img, ivec2(x, y)).xyz;

    imageStore(filter_img, ivec2(x, y), vec4(pixelValue / sum, 1.0));
}
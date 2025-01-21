/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 1) in;

layout(set = 0, binding = 0, rgba32f) uniform image2D filter_img;

layout( push_constant ) uniform constants {
    float pixels_per_degree;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(filter_img);

    if (x >= size.x || y >= size.y) {
        return;
    }

    vec3 positive_sum = vec3(0.0);
    vec3 negative_sum = vec3(0.0);

    for (int j = 0; j < size.x; j++) {
        for (int k = 0; k < size.y; k++) {
            vec3 value = imageLoad(filter_img, ivec2(j, k)).xyz;

            positive_sum += max(value, vec3(0.0));
            negative_sum += -min(value, vec3(0.0));
        }
    }

    vec3 pixelValue = imageLoad(filter_img, pos).xyz;

    imageStore(filter_img, pos, vec4(
        mix(pixelValue.x / negative_sum.x, pixelValue.x / positive_sum.x, pixelValue.x > 0.0),
        mix(pixelValue.y / negative_sum.y, pixelValue.y / positive_sum.y, pixelValue.y > 0.0),
        mix(pixelValue.z / negative_sum.z, pixelValue.z / positive_sum.z, pixelValue.z > 0.0),
        1.0
    ));
}
/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform image2D filter_img[2];

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

    float positive_sum = 0.0;
    float negative_sum = 0.0;

    for (int j = 0; j < size.x; j++) {
        for (int k = 0; k < size.y; k++) {
            float value = imageLoad(filter_img[z], ivec2(j, k)).x;
            if (value > 0) {
                positive_sum += value;
            } else {
                negative_sum += -value;
            }
        }
    }

    float pixelValue = imageLoad(filter_img[z], ivec2(x, y)).x;

    if (pixelValue > 0) {
        imageStore(filter_img[z], ivec2(x, y), vec4(pixelValue / positive_sum));
    } else {
        imageStore(filter_img[z], ivec2(x, y), vec4(pixelValue / negative_sum));
    }
}
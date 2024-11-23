/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 8, local_size_y = 8) in;

// each pixel is [I, Q, Y, 1], where I, Q, Y are FSIM color values
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D output_img;

const float verticalArray[9] = float[9](3.0, 0.0, -3.0, 10.0, 0.0, -10.0, 3.0, 0.0, -3.0);
const float horizontalArray[9] = float[9](3.0, 10.0, -3.0, 0.0, 0.0, 0.0, -3.0, -10.0, -3.0);

float verticalWeight(ivec2 pos) {
    int index = (pos.x + 1) + ((pos.y + 1) * 3);
    return verticalArray[index];
}

float horizontalWeight(ivec2 pos) {
    int index = (pos.x + 1) + ((pos.y + 1) * 3);
    return horizontalArray[index];
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 size = imageSize(input_img);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float vertSum = 0.0;
    float horSum = 0.0;

    for (int j = -1; j < 2; j++) {
        for (int k = -1; k < 2; k++) {
            int posX = int(x) + j;
            int posY = int(y) + k;
            if (posX >= 0 && posX < size.x && posY >= 0 && posY < size.y) {
                float inValue = imageLoad(input_img, ivec2(posX, posY)).z;
                vertSum += inValue * verticalWeight(ivec2(j, k));
                horSum += inValue * horizontalWeight(ivec2(j, k));
            }
        }
    }

    float total = sqrt(pow(vertSum / 16.0, 2.0) + pow(horSum / 16.0, 2.0));

    imageStore(output_img, pos, vec4(total, 0.0, 0.0, 0.0));
}
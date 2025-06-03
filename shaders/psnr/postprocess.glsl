/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 1, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InOutBuf {
    float data[];
};

layout( push_constant ) uniform constants {
    uint pixels;
} push_consts;

void main() {
    float inputVal = data[0] / float(push_consts.pixels);

    data[0] = -10.0 * (log2(inputVal) / log2(10.0));
}

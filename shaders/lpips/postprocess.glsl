/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 1, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer OutBuf {
    float data[];
};

layout( push_constant ) uniform constants {
    uint size;
} push_consts;

void main() {
    data[0] = data[0] / (push_consts.size);
}
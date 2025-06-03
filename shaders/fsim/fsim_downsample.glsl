/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 64, local_size_y = 1) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img[2];
// each pixel is [I, Q, Y, 1], where I, Q, Y are FSIM color values
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_img[2];

layout( push_constant ) uniform constants {
    // FSIM scaling factor
    int F;
} push_consts;

shared vec3 sums[64];

vec3 colorConvert(vec3 inColor) {
    return vec3(
        inColor.r * 0.596 - inColor.g * 0.274 - inColor.b * 0.322,
        inColor.r * 0.211 - inColor.g * 0.523 + inColor.b * 0.312,
        inColor.r * 0.299 + inColor.g * 0.587 + inColor.b * 0.114
    );
}

void main() {
    int tid = int(gl_LocalInvocationID.x);
    uint z = gl_WorkGroupID.z;
    ivec2 inSize = imageSize(input_img[z]);
    ivec2 size = imageSize(output_img[z]);

    uint x = gl_WorkGroupID.x % size.x;
    uint y = gl_WorkGroupID.x / size.x;

    sums[tid] = vec3(0);
    memoryBarrierShared();
    barrier();

    ivec2 pos = ivec2(x, y);

    for (uint offset = 0; offset < (push_consts.F * push_consts.F); offset += gl_WorkGroupSize.x) {
        int i = int(offset + tid);

        if (i >= (push_consts.F * push_consts.F)) {
            break;
        }

        int j = (int(x) * push_consts.F) - (push_consts.F / 2) + i % push_consts.F;
        int k = (int(y) * push_consts.F) - (push_consts.F / 2) + i / push_consts.F;

        if (j >= 0 && j < inSize.x && k >= 0 && k < inSize.y) {
            sums[tid] += colorConvert(imageLoad(input_img[z], ivec2(j, k)).xyz);
        }
    }

    memoryBarrierShared();
    barrier();

    if (tid < 32) {
        sums[tid] += sums[tid + 32];
    }
    memoryBarrierShared();
    barrier();
    if (tid < 16) {
        sums[tid] += sums[tid + 16];
    }
    memoryBarrierShared();
    barrier();
    if (tid < 8) {
        sums[tid] += sums[tid + 8];
    }
    memoryBarrierShared();
    barrier();
    if (tid < 4) {
        sums[tid] += sums[tid + 4];
    }
    memoryBarrierShared();
    barrier();
    if (tid < 2) {
        sums[tid] += sums[tid + 2];
    }
    memoryBarrierShared();
    barrier();
    if (tid < 1) {
        sums[tid] += sums[tid + 1];
    }
    memoryBarrierShared();
    barrier();

    float scaler = pow(push_consts.F, 2.0);
    if (gl_LocalInvocationID.x == 0) {
        imageStore(output_img[z], pos, vec4((sums[0] / scaler) * 255.0, 1.0));
    }
}
/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

#define SUM_SIZE 1024

layout (local_size_x = SUM_SIZE, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InOutBuf {
    float data[];
};

layout(std430, set = 0, binding = 1) buffer SortedBuf {
    float sortedData[];
};

layout( push_constant ) uniform constants {
    // number of elements to sum
    uint size;
    uint doMidDiff;
} push_consts;

shared float subSums[SUM_SIZE];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + tid;

    if (push_consts.doMidDiff != 0) {
        float median;
        if (push_consts.size % 2 == 0) {
            median = (sortedData[push_consts.size / 2 - 1] + sortedData[push_consts.size / 2]);
        } else {
            median = sortedData[push_consts.size / 2];
        }

        subSums[tid] = mix(abs(data[i] - median), 0.0, i >= push_consts.size);
    } else {
        subSums[tid] = mix(data[i], 0.0, i >= push_consts.size);
    }

    memoryBarrierShared();
    barrier();

    for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            subSums[tid] += subSums[tid + s];
        }

        memoryBarrierShared();
        barrier();
    }

    if (tid == 0) {
        data[gl_WorkGroupID.x] = subSums[0];
    }
}

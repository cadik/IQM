/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float inData[];
};

layout(std430, set = 0, binding = 1) buffer OutBuf {
    float outData[];
};

layout( push_constant ) uniform constants {
    uint width;
    uint height;
    uint targetWidth;
    uint targetHeight;
} push_consts;

const int KERNEL_SIZE = 3;
const int STRIDE = 2;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;

    if (x >= push_consts.targetWidth || y >= push_consts.targetHeight) {
        return;
    }

    uint channelSize = push_consts.width * push_consts.height;
    uint targetChannelSize = push_consts.targetWidth * push_consts.targetHeight;

    int srcCenterX = int(x) * STRIDE + 1;
    int srcCenterY = int(y) * STRIDE + 1;

    // is done after ReLU, so 0 should be min possible value
    float maxVal = 0;

    int halfOffset = KERNEL_SIZE / 2;
    for (int yOffset = -halfOffset; yOffset <= halfOffset; yOffset++) {
        for (int xOffset = -halfOffset; xOffset <= halfOffset; xOffset++) {
            int coord = (srcCenterX + xOffset) + int(push_consts.width) * (srcCenterY + yOffset);
            float val = inData[channelSize * z + coord];
            maxVal = max(maxVal, val);
        }
    }

    outData[targetChannelSize * z + (x + push_consts.targetWidth * y)] = maxVal;
}
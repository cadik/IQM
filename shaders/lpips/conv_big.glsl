/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (constant_id = 0) const int KERNEL_SIZE = 11;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
} inBufs[2];

layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
} outBufs[2];

layout(std430, set = 0, binding = 2) buffer WeightBuf {
    float weights[];
};

layout(std430, set = 0, binding = 3) buffer BiasBuf {
    float biases[];
};

layout( push_constant ) uniform constants {
    uint width;
    uint height;
    uint targetWidth;
    uint targetHeight;
    uint inChannels;
    uint kernelSize;
    uint padding;
    uint stride;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;

    if (x >= push_consts.targetWidth || y >= push_consts.targetHeight) {
        return;
    }

    float valueTotalTest = 0.0;
    float valueTotalRef = 0.0;

    int halfSize = KERNEL_SIZE / 2;
    int srcCenterX = int(x * push_consts.stride + halfSize) - int(push_consts.padding);
    int srcCenterY = int(y * push_consts.stride + halfSize) - int(push_consts.padding);

    uint channelSize = push_consts.width * push_consts.height;
    uint targetChannelSize = push_consts.targetWidth * push_consts.targetHeight;

    // center pixels no not have to worry about accessing out of bounds data
    if ((srcCenterX - halfSize) < 0 || ((srcCenterX + halfSize + 1) >= push_consts.width) || (srcCenterY - halfSize) <= 0 || ((srcCenterY + halfSize + 1) >= push_consts.height)) {
        for (int i = 0; i < push_consts.inChannels; i++) {
            int inChannelOffset = i * int(KERNEL_SIZE * KERNEL_SIZE * gl_NumWorkGroups.z);

            for (int y = -halfSize; y <= halfSize; y++) {
                int srcY = srcCenterY + y;
                for (int x = -halfSize; x <= halfSize; x++) {
                    int posOffset = (KERNEL_SIZE * (y + halfSize) + (x + halfSize)) * int(gl_NumWorkGroups.z);

                    int srcX = srcCenterX + x;

                    float valueTest = 0.0;
                    float valueRef = 0.0;
                    if (srcX >= 0 && srcX < push_consts.width && srcY >= 0 && srcY < push_consts.height) {
                        valueTest = inBufs[0].data[channelSize * i + srcX + push_consts.width * srcY];
                        valueRef = inBufs[1].data[channelSize * i + srcX + push_consts.width * srcY];
                    }

                    float weight = weights[inChannelOffset + posOffset + z];

                    valueTotalTest += valueTest * weight;
                    valueTotalRef += valueRef * weight;
                }
            }
        }
    } else {
        for (int i = 0; i < push_consts.inChannels; i++) {
            int inChannelOffset = i * int(KERNEL_SIZE * KERNEL_SIZE * gl_NumWorkGroups.z);

            for (int y = -halfSize; y <= halfSize; y++) {
                int srcY = srcCenterY + y;
                for (int x = -halfSize; x <= halfSize; x++) {
                    int posOffset = (KERNEL_SIZE * (y + halfSize) + (x + halfSize)) * int(gl_NumWorkGroups.z);

                    int srcX = srcCenterX + x;

                    float valueTest = inBufs[0].data[channelSize * i + srcX + push_consts.width * srcY];
                    float valueRef = inBufs[1].data[channelSize * i + srcX + push_consts.width * srcY];

                    float weight = weights[inChannelOffset + posOffset + z];

                    valueTotalTest += valueTest * weight;
                    valueTotalRef += valueRef * weight;
                }
            }
        }
    }

    float reluTest = max(0.0, valueTotalTest + biases[z]);
    float reluRef = max(0.0, valueTotalRef + biases[z]);
    outBufs[0].data[targetChannelSize * z + x + push_consts.targetWidth * y] = reluTest;
    outBufs[1].data[targetChannelSize * z + x + push_consts.targetWidth * y] = reluRef;
}
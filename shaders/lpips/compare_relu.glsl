/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) buffer TestBuf {
    float testData[];
};

layout(std430, set = 0, binding = 1) buffer RefBuf {
    float refData[];
};

layout(std430, set = 0, binding = 2) buffer OutBuf {
    float outData[];
};

layout(std430, set = 0, binding = 3) buffer WeightBuf {
    float weights[];
};

layout( push_constant ) uniform constants {
    uint width;
    uint height;
    uint channels;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    if (x >= push_consts.width || y >= push_consts.height) {
        return;
    }

    uint channelSize = push_consts.width * push_consts.height;

    float value = 0.0;
    float sumTest = 0.00001;
    float sumRef = 0.00001;

    for (int i = 0; i < push_consts.channels; i++) {
        float test = testData[channelSize * i + x + push_consts.width * y];
        float ref = refData[channelSize * i + x + push_consts.width * y];

        sumTest += pow(test, 2.0);
        sumRef += pow(ref, 2.0);
    }

    sumTest = sqrt(sumTest);
    sumRef = sqrt(sumRef);

    for (int i = 0; i < push_consts.channels; i++) {
        float test = testData[channelSize * i + x + push_consts.width * y] / sumTest;
        float ref = refData[channelSize * i + x + push_consts.width * y] / sumRef;

        float delta = pow(test - ref, 2.0);

        value += delta * weights[i];
    }

    outData[x + push_consts.width * y] = value;
}
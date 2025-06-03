/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 1024, local_size_y = 1) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
} inData[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
} outData;

layout(set = 0, binding = 2, rgba32f) uniform readonly image2D filter_img;

layout( push_constant ) uniform constants {
    uint size;
    uint width;
    uint height;
} push_consts;

void main() {
    uint pixel = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (pixel >= push_consts.size) {
        return;
    }

    uint x = pixel % push_consts.width;
    uint y = pixel / push_consts.width;
    ivec2 pos = ivec2(x, y);
    ivec2 filterSize = imageSize(filter_img);

    float qf = 0.5;

    vec2 dInput = vec2(0.0);
    vec2 dRef = vec2(0.0);
    vec2 ddInput = vec2(0.0);
    vec2 ddRef = vec2(0.0);

    for (int k = -filterSize.x / 2; k <= filterSize.x/2; k++) {
        uint actualY = uint(clamp(int(y) - k, 0, int(push_consts.height) - 1));
        uint filterY = k + filterSize.x / 2;

        uint index = (x + actualY * push_consts.width) * 3;

        vec3 valueInput = vec3(inData[0].data[index], inData[0].data[index + 1], inData[0].data[index + 2]);
        vec3 valueRef = vec3(inData[1].data[index], inData[1].data[index + 1], inData[1].data[index + 2]);

        vec3 filterWeights = imageLoad(filter_img, ivec2(filterY, 0)).xyz;

        dInput += vec2(valueInput.x * filterWeights.x, valueInput.z * filterWeights.y);
        dRef += vec2(valueRef.x * filterWeights.x, valueRef.z * filterWeights.y);
        ddInput += vec2(valueInput.y * filterWeights.x, valueInput.z * filterWeights.z);
        ddRef += vec2(valueRef.y * filterWeights.x, valueRef.z * filterWeights.z);
    }

    float edgeDiff = abs(length(dInput) - length(dRef));
    float pointDiff = abs(length(ddInput) - length(ddRef));

    float scaler = inversesqrt(2.0);
    float diff = max(edgeDiff, pointDiff);
    float deltaEf = pow(scaler * diff, qf);

    outData.data[pixel] = deltaEf;
}
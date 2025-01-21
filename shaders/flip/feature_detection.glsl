/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, r32f) uniform writeonly image2D output_img;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D filter_img;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[0]);
    ivec2 filterSize = imageSize(filter_img);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float qf = 0.5;

    vec2 dInput = vec2(0.0);
    vec2 dRef = vec2(0.0);
    vec2 ddInput = vec2(0.0);
    vec2 ddRef = vec2(0.0);

    for (int k = -filterSize.x / 2; k <= filterSize.x/2; k++) {
        uint actualY = uint(clamp(int(y) - k, 0, size.y - 1));
        uint filterY = k + filterSize.x / 2;

        vec3 valueInput = imageLoad(input_img[0], ivec2(x, actualY)).xyz;
        vec3 valueRef = imageLoad(input_img[1], ivec2(x, actualY)).xyz;

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

    imageStore(output_img, pos, vec4(deltaEf));
}
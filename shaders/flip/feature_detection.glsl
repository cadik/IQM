/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_img[2];
layout(set = 0, binding = 1, r32f) uniform readonly image2D filter_img[2];
layout(set = 0, binding = 2, r32f) uniform writeonly image2D output_img;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[0]);
    ivec2 filterSize = imageSize(filter_img[0]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float qf = 0.5;

    vec2 edgeInput = vec2(0.0);
    vec2 edgeRef = vec2(0.0);
    vec2 pointInput = vec2(0.0);
    vec2 pointRef = vec2(0.0);

    for (int j = -filterSize.x / 2; j <= filterSize.x/2; j++) {
        uint actualX = uint(clamp(int(x) - j, 0, size.x - 1));
        uint filterX = j + filterSize.x / 2;

        for (int k = -filterSize.y / 2; k <= filterSize.y/2; k++) {
            uint actualY = uint(clamp(int(y) - k, 0, size.y - 1));
            uint filterY = k + filterSize.y / 2;

            float valueInput = (imageLoad(input_img[0], ivec2(actualX, actualY)).x + 16.0) / 116.0;
            float valueRef = (imageLoad(input_img[1], ivec2(actualX, actualY)).x + 16.0) / 116.0;

            float filterWeightEdgeX = imageLoad(filter_img[0], ivec2(filterX, filterY)).x;
            float filterWeightEdgeY = imageLoad(filter_img[0], ivec2(filterY, filterX)).x;
            float filterWeightPointX = imageLoad(filter_img[1], ivec2(filterX, filterY)).x;
            float filterWeightPointY = imageLoad(filter_img[1], ivec2(filterY, filterX)).x;

            edgeInput.x += valueInput * filterWeightEdgeX;
            edgeInput.y += valueInput * filterWeightEdgeY;
            edgeRef.x += valueRef * filterWeightEdgeX;
            edgeRef.y += valueRef * filterWeightEdgeY;
            pointInput.x += valueInput * filterWeightPointX;
            pointInput.y += valueInput * filterWeightPointY;
            pointRef.x += valueRef * filterWeightPointX;
            pointRef.y += valueRef * filterWeightPointY;
        }
    }

    float edgeDiff = abs(length(edgeInput) - length(edgeRef));
    float pointDiff = abs(length(pointInput) - length(pointRef));

    float scaler = inversesqrt(2.0);
    float diff = max(edgeDiff, pointDiff);
    float deltaEf = pow(scaler * diff, qf);

    imageStore(output_img, pos, vec4(deltaEf));
}
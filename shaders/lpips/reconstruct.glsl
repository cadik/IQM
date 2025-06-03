/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) buffer InBuf {
    float data[];
};
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float outData[];
};

layout( push_constant ) uniform constants {
    uint width;
    uint height;
    uint width0;
    uint height0;
    uint width1;
    uint height1;
    uint width2;
    uint height2;
} push_consts;

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);
    ivec2 size = ivec2(push_consts.width, push_consts.height);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float value = 0;
    uint offset = 0;

    float xNorm = (float(x) + 0.5) * (float(push_consts.width0) / float(size.x)) - 0.5;
    float yNorm = (float(y) + 0.5) * (float(push_consts.height0) / float(size.y)) - 0.5;

    int nearestX = max(int(floor(xNorm)), 0);
    int nearestTopX = min(int(ceil(xNorm)), int(push_consts.width0 - 1));
    float ratio = fract(xNorm);
    int nearestY = max(int(floor(yNorm)), 0);
    int nearestTopY = min(int(ceil(yNorm)), int(push_consts.height0 - 1));
    float ratioY = fract(yNorm);

    float xNorm1 = (float(x) + 0.5) * (float(push_consts.width1) / float(size.x)) - 0.5;
    float yNorm1 = (float(y) + 0.5) * (float(push_consts.height1) / float(size.y)) - 0.5;

    int nearestX1 = max(int(floor(xNorm1)), 0);
    int nearestTopX1 = min(int(ceil(xNorm1)), int(push_consts.width1 - 1));
    float ratio1 = fract(xNorm1);
    int nearestY1 = max(int(floor(yNorm1)), 0);
    int nearestTopY1 = min(int(ceil(yNorm1)), int(push_consts.height1 - 1));
    float ratioY1 = fract(yNorm1);

    float xNorm2 = (float(x) + 0.5) * (float(push_consts.width2) / float(size.x)) - 0.5;
    float yNorm2 = (float(y) + 0.5) * (float(push_consts.height2) / float(size.y)) - 0.5;

    int nearestX2 = max(int(floor(xNorm2)), 0);
    int nearestTopX2 = min(int(ceil(xNorm2)), int(push_consts.width2 - 1));
    float ratio2 = fract(xNorm2);
    int nearestY2 = max(int(floor(yNorm2)), 0);
    int nearestTopY2 = min(int(ceil(yNorm2)), int(push_consts.height2 - 1));
    float ratioY2 = fract(yNorm2);

    float valueLeft = data[nearestX + push_consts.width0 * nearestY];
    float valueRight = data[nearestTopX + push_consts.width0 * nearestY];
    float valueTopLeft = data[nearestX + push_consts.width0 * nearestTopY];
    float valueTopRight = data[nearestTopX + push_consts.width0 * nearestTopY];

    float valueMixed = mix(valueLeft, valueRight, ratio);
    float valueMixedTop = mix(valueTopLeft, valueTopRight, ratio);
    value += mix(valueMixed, valueMixedTop, ratioY);
    offset += push_consts.width0 * push_consts.height0;

    valueLeft = data[offset + nearestX1 + push_consts.width1 * nearestY1];
    valueRight = data[offset + nearestTopX1 + push_consts.width1 * nearestY1];
    valueTopLeft = data[offset + nearestX1 + push_consts.width1 * nearestTopY1];
    valueTopRight = data[offset + nearestTopX1 + push_consts.width1 * nearestTopY1];

    valueMixed = mix(valueLeft, valueRight, ratio1);
    valueMixedTop = mix(valueTopLeft, valueTopRight, ratio1);
    value += mix(valueMixed, valueMixedTop, ratioY1);
    offset += push_consts.width1 * push_consts.height1;

    valueLeft = data[offset + nearestX2 + push_consts.width2 * nearestY2];
    valueRight = data[offset + nearestTopX2 + push_consts.width2 * nearestY2];
    valueTopLeft = data[offset + nearestX2 + push_consts.width2 * nearestTopY2];
    valueTopRight = data[offset + nearestTopX2 + push_consts.width2 * nearestTopY2];

    valueMixed = mix(valueLeft, valueRight, ratio2);
    valueMixedTop = mix(valueTopLeft, valueTopRight, ratio2);
    value += mix(valueMixed, valueMixedTop, ratioY2);
    offset += push_consts.width2 * push_consts.height2;

    valueLeft = data[offset + nearestX2 + push_consts.width2 * nearestY2];
    valueRight = data[offset + nearestTopX2 + push_consts.width2 * nearestY2];
    valueTopLeft = data[offset + nearestX2 + push_consts.width2 * nearestTopY2];
    valueTopRight = data[offset + nearestTopX2 + push_consts.width2 * nearestTopY2];

    valueMixed = mix(valueLeft, valueRight, ratio2);
    valueMixedTop = mix(valueTopLeft, valueTopRight, ratio2);
    value += mix(valueMixed, valueMixedTop, ratioY2);
    offset += push_consts.width2 * push_consts.height2;

    valueLeft = data[offset + nearestX2 + push_consts.width2 * nearestY2];
    valueRight = data[offset + nearestTopX2 + push_consts.width2 * nearestY2];
    valueTopLeft = data[offset + nearestX2 + push_consts.width2 * nearestTopY2];
    valueTopRight = data[offset + nearestTopX2 + push_consts.width2 * nearestTopY2];

    valueMixed = mix(valueLeft, valueRight, ratio2);
    valueMixedTop = mix(valueTopLeft, valueTopRight, ratio2);
    value += mix(valueMixed, valueMixedTop, ratioY2);

    outData[pos.x + size.x * pos.y] = value;
}
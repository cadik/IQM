#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D ref_img;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_img;

layout( push_constant ) uniform constants {
    int kernelSize;
    float k_1;
    float k_2;
    float sigma;
} push_consts;

// Rec. 601
float luminance(vec3 color) {
    return 0.2989 * color.r + 0.5810 * color.g + 0.1140 * color.b;
}

float gaussWeight(int offset) {
    return (1.0 / (push_consts.sigma * pow(2.0 * 3.1415, 0.5))) * pow(2.71, -0.5 * pow(offset / push_consts.sigma, 2.0));
}

float gaussBlur(ivec2 pos) {
    ivec2 maxPos = imageSize(input_img);

    float total = 0.0;
    float totalWeight = 0.0;

    int start = -(push_consts.kernelSize - 1) / 2;
    int end = (push_consts.kernelSize - 1) / 2;

    for (int xOffset = start; xOffset <= end; xOffset++) {
        for (int yOffset = start; yOffset <= end; yOffset++) {
            int x = pos.x + xOffset;
            int y = pos.y + yOffset;

            if (x >= maxPos.x || y >= maxPos.y || x < 0 || y < 0) {
                continue;
            }

            float weight = gaussWeight(xOffset) * gaussWeight(yOffset);
            total += luminance(imageLoad(input_img, ivec2(x, y)).xyz) * weight;
            totalWeight += weight;
        }
    }

    return total / totalWeight;
}

void main() {
    float sigma = 1.5;

    float c_1 = pow(push_consts.k_1, 2);
    float c_2 = pow(push_consts.k_2, 2);

    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 pos = ivec2(x, y);

    if (x > imageSize(input_img).x || y > imageSize(input_img).y) {
        return;
    }

    float outCol = 1.0 - abs(gaussBlur(pos) - luminance(imageLoad(ref_img, ivec2(x, y)).xyz));

    imageStore(output_img, pos, vec4(vec3(outCol), 1.0));
}
#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img;
// each pixel is [I, Q, Y, 1], where I, Q, Y are FSIM color values
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D output_img;

layout( push_constant ) uniform constants {
    // FSIM scaling factor
    int F;
    // bool passed as int
    int colorConvert;
} push_consts;

vec4 colorConvert(vec4 inColor) {
    if (push_consts.colorConvert != 0) {
        return vec4(
            inColor.r * 0.596 - inColor.g * 0.274 - inColor.b * 0.322,
            inColor.r * 0.211 - inColor.g * 0.523 + inColor.b * 0.312,
            inColor.r * 0.299 + inColor.g * 0.587 + inColor.b * 0.114,
            1.0
        );
    } else {
        return vec4(1.0, 1.0, inColor.r, 1.0);
    }
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 size = imageSize(input_img);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    vec4 sum = vec4(0.0);
    float scaler = pow(push_consts.F, 2.0);

    int xStart = (int(x) * push_consts.F) - (push_consts.F / 2);
    int yStart = (int(y) * push_consts.F) - (push_consts.F / 2);

    for (int j = xStart; j < (xStart + push_consts.F); j++) {
        for (int k = yStart; k < (yStart + push_consts.F); k++) {
            if (j >= 0 && j < size.x && k >= 0 && k < size.y) {
                sum += colorConvert(imageLoad(input_img, ivec2(j, k)));
            }
        }
    }

    imageStore(output_img, pos, sum / scaler);
}
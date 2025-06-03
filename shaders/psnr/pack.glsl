/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D input_img[2];
layout(std430, set = 0, binding = 1) buffer OutBuf {
    float data[];
};

layout( push_constant ) uniform constants {
    int variant;
} push_consts;

// Rec. 601 - same as openCV
float luminance(vec4 color) {
    return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

vec3 rec601(vec3 color) {
    // map luma to 0-1
    return vec3(
               0.299    * color.r + 0.587     * color.g + 0.114    * color.b,
       0.5 + - 0.168736 * color.r - 0.331264  * color.g + 0.5      * color.b,
       0.5 +   0.5      * color.r - 0.0418688 * color.g - 0.081312 * color.b
    );
}

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 maxPos = imageSize(input_img[0]);
    ivec2 pos = ivec2(x, y);

    if (x >= maxPos.x || y >= maxPos.y) {
        return;
    }

    vec4 inpTest = imageLoad(input_img[0], pos);
    vec4 inpRef = imageLoad(input_img[1], pos);

    uint index = x + maxPos.x * y;

    float value;

    if (push_consts.variant == 0) {
        // luma
        value = pow(luminance(inpTest) - luminance(inpRef), 2.0);
    } else if (push_consts.variant == 1) {
        // rgb
        vec3 difSq = pow(inpTest.rgb - inpRef.rgb, vec3(2.0));
        value = dot(difSq, vec3(1.0)) / 3.0;
    } else if (push_consts.variant == 2) {
        // ycbcr
        vec3 difSq = pow(rec601(inpTest.rgb) - rec601(inpRef.rgb), vec3(2.0));
        value = dot(difSq, vec3(1.0)) / 3.0;
    }

    data[index] = value;
}
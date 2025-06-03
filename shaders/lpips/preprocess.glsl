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
} outputs[2];

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(input_img[z]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    vec3 color = imageLoad(input_img[z], pos).rgb;

    // rescale to -1;1
    color -= 0.5;
    color *= 2.0;

    // now resacle according to LPIPS
    color += vec3(0.03, 0.088, 0.188);
    color /= vec3(0.458, 0.448, 0.450);

    uint index = (x + size.x * y);
    uint offset = (size.x * size.y);

    outputs[z].data[index] = color.r;
    outputs[z].data[index + offset] = color.g;
    outputs[z].data[index + offset * 2] = color.b;
}
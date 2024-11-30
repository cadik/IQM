/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

layout (local_size_x = 8, local_size_y = 8) in;

// each pixel is [I, Q, Y, 1], where I, Q, Y are FSIM color values
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_imgs[2];
layout(set = 0, binding = 1, r32f) uniform readonly image2D gradient_imgs[2];
layout(set = 0, binding = 2, r32f) uniform readonly image2D phase_congruency_imgs[2];
// The three output images need to be summed separately
layout(set = 0, binding = 3, r32f) uniform writeonly image2D output_imgs[3];

void main() {
    float T1 = 0.85;
    float T2 = 160;
    float T3 = 200;
    float T4 = 200;
    float lambda = 0.03;

    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;

    ivec2 size = imageSize(input_imgs[0]);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float pc1 = imageLoad(phase_congruency_imgs[0], pos).x;
    float pc2 = imageLoad(phase_congruency_imgs[1], pos).x;
    float pcm = max(pc1, pc2);
    imageStore(output_imgs[0], pos, vec4(pcm));

    float pcsim = (2.0 * pc1 * pc2 + T1) / (pow(pc1, 2.0) + pow(pc2, 2.0) + T1);

    float g1 = imageLoad(gradient_imgs[0], pos).x;
    float g2 = imageLoad(gradient_imgs[1], pos).x;

    float gradsim = (2.0 * g1 * g2 + T2) / (pow(g1, 2.0) + pow(g2, 2.0) + T2);

    float sim = gradsim * pcsim * pcm;
    imageStore(output_imgs[1], pos, vec4(sim));

    vec2 iq1 = imageLoad(input_imgs[0], pos).xy;
    vec2 iq2 = imageLoad(input_imgs[1], pos).xy;

    float isim = (2.0 * iq1.x * iq2.x + T3) / (pow(iq1.x, 2.0) + pow(iq2.x, 2.0) + T3);
    float qsim = (2.0 * iq1.y * iq2.y + T4) / (pow(iq1.y, 2.0) + pow(iq2.y, 2.0) + T4);

    float simc = gradsim * pcsim * pow(abs(isim * qsim), lambda) * pcm;
    imageStore(output_imgs[2], pos, vec4(simc));
}
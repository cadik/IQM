/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define PI 3.141592653589
#define ORIENTATIONS 4
#define SCALES 4
#define OxS 16

layout (local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D phase_congruency[2];
layout(std430, set = 0, binding = 1) buffer InNPBuf {
    float noisePowers[];
};
layout(std430, set = 0, binding = 2) buffer InEnergyEstBufs {
    float est;
} energyEsts[8];
// each pixel is [Am, Energy, 0, 0]
layout(set = 0, binding = 3, rg32f) uniform readonly image2D filter_responses[8];

layout( push_constant ) uniform constants {
    uint index;
} push_consts;

void main() {
    float k = 2.0;

    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    ivec2 size = imageSize(phase_congruency[push_consts.index]);
    ivec2 pos = ivec2(x, y);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float energyTotal = 0.0;
    float ampTotal = 0.0;

    for (int o = 0; o < ORIENTATIONS; o++) {
        float noisePower = noisePowers[o + ORIENTATIONS * push_consts.index];
        float sumEstSumAn2 = energyEsts[2 * o].est;
        float sumEstSumAiAj = energyEsts[2 * o + 1].est;
        float estNoiseEnergy2 = 2.0 * noisePower * sumEstSumAn2 + 4.0 * noisePower * sumEstSumAiAj;

        float tau = sqrt(estNoiseEnergy2 / 2.0);
        float estNoiseEnergy = tau * sqrt(PI / 2);
        float estNoiseEnergySigma = sqrt((2 - PI/2) * pow(tau, 2.0));

        float T = estNoiseEnergy + k * estNoiseEnergySigma;
        T = T/1.7;

        vec2 amp_energy = imageLoad(filter_responses[o + push_consts.index * ORIENTATIONS], pos).xy;

        float energy = max(amp_energy.y - T, 0);
        energyTotal += energy;
        ampTotal += amp_energy.x;
    }

    float val = energyTotal / ampTotal;

    imageStore(phase_congruency[push_consts.index], pos, vec4(val));
}
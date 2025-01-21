/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#version 450
#pragma shader_stage(compute)

#define PI 3.141592653589
#define ORIENTATIONS 4

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r32f) uniform writeonly image2D out_filter[ORIENTATIONS];

void main() {
    uint x = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint y = gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y;
    uint z = gl_WorkGroupID.z;
    ivec2 pos = ivec2(x, y);
    ivec2 size = imageSize(out_filter[z]);

    if (x >= size.x || y >= size.y) {
        return;
    }

    float dThetaOnSigma = 1.2;
    float thetaSigma = PI / ORIENTATIONS / dThetaOnSigma;

    // do the equivalent of ifftshift first, so writes are ordered
    uint shiftedX = (x + uint(size.x)/2) % size.x;
    uint shiftedY = (y + uint(size.y)/2) % size.y;
    float scaledX = float(int(shiftedX) - size.x/2) / float(size.x);
    float scaledY = float(int(shiftedY) - size.y/2) / float(size.y);

    float theta = atan(-scaledY, scaledX);
    float sintheta = sin(theta);
    float costheta = cos(theta);

    float angle = ((float(z) * PI) / ORIENTATIONS);
    float ds = sintheta * cos(angle) - costheta * sin(angle);
    float dc = costheta * cos(angle) + sintheta * sin(angle);
    float dtheta = abs(atan(ds, dc));

    float value = exp(-pow(dtheta, 2.0) / (2.0 * pow(thetaSigma, 2.0)));

    imageStore(out_filter[z], pos, vec4(value));
}
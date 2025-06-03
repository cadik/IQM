/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#define E 2.71828182846
#define PI 3.141592653589

float gaussWeight(int offset, float sigma) {
    float dist = offset * offset;
    return pow(E, -(dist / (2.0 * pow(sigma, 2.0))));
}

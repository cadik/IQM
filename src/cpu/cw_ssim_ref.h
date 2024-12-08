/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef CW_SSIM_REF_H
#define CW_SSIM_REF_H

namespace IQM::CPU {
    struct CW_SSIMInputImage {
        void * data;
        unsigned int width;
        unsigned int height;
    };

    class CW_SSIM_Ref {
    public:
        void computeMetric(const CW_SSIMInputImage &input, const CW_SSIMInputImage &reference) const;

        float k = 0.0;
    };
}

#endif
#ifndef CW_SSIM_REF_H
#define CW_SSIM_REF_H

#include <opencv2/opencv.hpp>

namespace IQM::CPU {
    class CW_SSIM_Ref {
    public:
        [[nodiscard]] cv::Mat computeMetric(const cv::Mat &input, const cv::Mat &reference) const;

        float k = 0.0;
    };
}

#endif
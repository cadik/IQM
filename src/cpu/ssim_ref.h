#ifndef IQM_SSIM_REF_H
#define IQM_SSIM_REF_H

#include <opencv2/opencv.hpp>

namespace IQM::CPU {
    class SSIM_Reference {
    public:
        [[nodiscard]] cv::Mat computeMetric(const cv::Mat &input, const cv::Mat &reference) const;
        int kernelSize = 11;
        float k_1 = 0.01;
        float k_2 = 0.03;
        float sigma = 1.5;
    };
}

#endif //IQM_SSIM_REF_H

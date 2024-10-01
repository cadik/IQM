#ifndef IQM_SSIM_REF_H
#define IQM_SSIM_REF_H

#include <opencv2/opencv.hpp>

namespace IQM::CPU {
    class SSIM_Reference {
    public:
        cv::Mat computeMetric(cv::Mat input, cv::Mat reference);
        int kernelSize = 11;
    };
}

#endif //IQM_SSIM_REF_H

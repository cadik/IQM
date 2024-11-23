/*
* Image Quality Metrics
 * Petr Volf - 2024
 */

#include "cw_ssim_ref.h"

cv::Mat IQM::CPU::CW_SSIM_Ref::computeMetric(const cv::Mat &input, const cv::Mat &reference) const {
    cv::Mat greyInput;
    cv::cvtColor(input, greyInput, cv::COLOR_BGR2GRAY);
    cv::Mat inputFloat;
    greyInput.convertTo(inputFloat, CV_32F);

    cv::Mat out;

    cv::dft(inputFloat, out);

    return out;
}

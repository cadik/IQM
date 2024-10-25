#include "ssim_ref.h"

cv::Mat IQM::CPU::SSIM_Reference::computeMetric(const cv::Mat& input, const cv::Mat& reference) const {
    auto kernel = cv::Size{this->kernelSize, this->kernelSize};
    auto l = 1.0;
    auto sigma = 1.5;

    auto c_1 = pow(this->k_1 * l, 2);
    auto c_2 = pow(this->k_2 * l, 2);

    cv::Mat greyInput;
    cv::cvtColor(input, greyInput, cv::COLOR_BGR2GRAY);
    cv::Mat greyRef;
    cv::cvtColor(reference, greyRef, cv::COLOR_BGR2GRAY);
    cv::Mat inputFloat;
    greyInput.convertTo(inputFloat, CV_32F);
    cv::Mat refFloat;
    greyRef.convertTo(refFloat, CV_32F);

    cv::Mat blurredInput;
    cv::GaussianBlur(inputFloat, blurredInput, kernel, sigma);

    cv::Mat blurredRef;
    cv::GaussianBlur(refFloat, blurredRef, kernel, sigma);

    cv::Mat varianceInput = inputFloat - blurredInput;
    cv::Mat varianceRef = refFloat - blurredRef;

    cv::Mat contravariance;
    cv::multiply(varianceInput, varianceRef, contravariance);

    varianceInput.forEach<float>([](float& i, const int []) {i *= i;});
    cv::Mat varianceInputBlurred;
    cv::GaussianBlur(varianceInput, varianceInputBlurred, kernel, sigma);

    varianceRef.forEach<float>([](float& i, const int []) {i *= i;});
    cv::Mat varianceRefBlurred;
    cv::GaussianBlur(varianceRef, varianceRefBlurred, kernel, sigma);

    cv::Mat contravarianceRefBlurred;
    cv::GaussianBlur(contravariance, contravarianceRefBlurred, kernel, sigma);

    cv::Mat out = input;
    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
            auto mean_img = blurredInput.at<float>(y, x);
            auto mean_ref = blurredRef.at<float>(y, x);

            auto variance_img = varianceInputBlurred.at<float>(y, x);
            auto variance_ref = varianceRefBlurred.at<float>(y, x);
            auto contravariance_pixel = contravarianceRefBlurred.at<float>(y, x);

            auto val = ((2.0 * mean_img * mean_ref + c_1) * (2.0 * contravariance_pixel + c_2)) /
                    ((pow(mean_img, 2.0) + pow(mean_ref, 2.0) + c_1) * (variance_img + variance_ref + c_2));
            auto val_mapped = static_cast<uchar>(std::max(val * 255.0, 0.0));

            auto col = cv::Vec3b{val_mapped, val_mapped, val_mapped};
            out.at<cv::Vec3b>(y, x) = col;
        }
    }

    return out;
}

#include "ssim_ref.h"

cv::Mat IQM::CPU::SSIM_Reference::computeMetric(cv::Mat input, cv::Mat reference) {
    auto kernel = cv::Size{this->kernelSize, this->kernelSize};
    auto l = 255.0;
    auto sigma = 1.5;

    auto c_1 = pow(this->k_1 * l, 2);
    auto c_2 = pow(this->k_2 * l, 2);

    cv::Mat blurredInput;
    cv::GaussianBlur(input, blurredInput, kernel, sigma);

    cv::Mat blurredRef;
    cv::GaussianBlur(reference, blurredRef, kernel, sigma);

    cv::Mat varianceInput = input - blurredInput;
    cv::Mat varianceRef = reference - blurredRef;

    cv::Mat cotravariance;
    cv::multiply(varianceInput, varianceRef, cotravariance);

    varianceInput.forEach<cv::Vec3b>([](cv::Vec3b& i, const int []) {i[0] *= i[0];i[1] *= i[1];i[2] *= i[2];});
    cv::Mat varianceInputBlurred;
    cv::GaussianBlur(varianceInput, varianceInputBlurred, kernel, sigma);

    varianceRef.forEach<cv::Vec3b>([](cv::Vec3b& i, const int []) {i[0] *= i[0];i[1] *= i[1];i[2] *= i[2];});
    cv::Mat varianceRefBlurred;
    cv::GaussianBlur(varianceRef, varianceRefBlurred, kernel, sigma);

    cv::Mat contravarianceRefBlurred;
    cv::GaussianBlur(cotravariance, contravarianceRefBlurred, kernel, sigma);

    cv::Mat out = input;
    for (unsigned y = 0; y < (unsigned)input.rows; y++) {
        for (unsigned x = 0; x < (unsigned)input.cols; x++) {
            auto mean_img = blurredInput.at<cv::Vec3b>((int)y, (int)x);
            auto mean_ref = blurredRef.at<cv::Vec3b>((int)y, (int)x);

            auto variance_img = varianceInputBlurred.at<cv::Vec3b>((int)y, (int)x);
            auto variance_ref = varianceRefBlurred.at<cv::Vec3b>((int)y, (int)x);
            auto contravariance_pixel = varianceRefBlurred.at<cv::Vec3b>((int)y, (int)x);

            auto val = ((2.0 * mean_img[0] * mean_ref[0] + c_1) * (2.0 * contravariance_pixel[0] + c_2)) /
                    ((pow(mean_img[0], 2) + pow(mean_ref[0], 2) + c_1) * (variance_img[0] + variance_ref[0] + c_2));
            auto val_mapped = (uchar)(val * 255.0);

            auto col = cv::Vec3b{val_mapped, val_mapped, val_mapped};
            out.at<cv::Vec3b>((int)y, (int)x) = col;
        }
    }

    return out;
}

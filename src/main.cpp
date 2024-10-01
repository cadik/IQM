#include <iostream>
#include "args.h"
#include "cpu/ssim_ref.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    auto args = IQM::Args(argc, argv);
    std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;

    cv::Mat image = imread(args.input_path, cv::IMREAD_COLOR);
    cv::Mat ref = imread(args.ref_path, cv::IMREAD_COLOR);

    auto method = IQM::CPU::SSIM_Reference();
    auto out = method.computeMetric(image, ref);

    namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", out);

    cv::waitKey(0);
    return 0;
}

#include <iostream>
#include "args.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    auto args = IQM::Args(argc, argv);
    std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;

    cv::Mat image = imread(args.input_path, cv::IMREAD_COLOR);
    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size{45, 45}, 0.0);

    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    imshow("Display Image", blurred);

    cv::waitKey(0);
    return 0;
}

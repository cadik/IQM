#include <iostream>
#include "args.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    auto args = IQM::Args(argc, argv);
    std::cout << "Selected method: " << IQM::method_name(args.method) << std::endl;

    cv::Mat image;
    image = imread( args.input_path, cv::IMREAD_COLOR );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    cv::waitKey(0);
    return 0;
}

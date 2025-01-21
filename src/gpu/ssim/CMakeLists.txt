cmake_minimum_required(VERSION 3.29)
project(IQM-SSIM)

add_library(IQM-SSIM STATIC ssim.cpp)

target_include_directories(IQM-SSIM
        PUBLIC "${PROJECT_SOURCE_DIR}"
)
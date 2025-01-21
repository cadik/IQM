cmake_minimum_required(VERSION 3.29)
project(IQM-SVD)

add_library(IQM-SVD STATIC svd.cpp)

target_include_directories(IQM-SVD
        PUBLIC "${PROJECT_SOURCE_DIR}"
)
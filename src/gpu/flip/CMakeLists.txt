cmake_minimum_required(VERSION 3.29)
project(IQM-FLIP)

add_library(IQM-FLIP STATIC flip.cpp
)

target_include_directories(IQM-FLIP
        PUBLIC "${PROJECT_SOURCE_DIR}"
)
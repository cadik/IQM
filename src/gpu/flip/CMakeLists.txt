cmake_minimum_required(VERSION 3.29)
project(IQM-FLIP)

add_library(IQM-FLIP STATIC flip.cpp
        flip_color_pipeline.cpp
        flip_color_pipeline.h
)

target_include_directories(IQM-FLIP
        PUBLIC "${PROJECT_SOURCE_DIR}"
)
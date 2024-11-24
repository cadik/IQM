cmake_minimum_required(VERSION 3.29)
project(IQM-FSIM)

add_library(IQM-FSIM STATIC fsim.cpp
        steps/fsim_lowpass_filter.cpp
        steps/fsim_lowpass_filter.h)
target_include_directories(IQM-FSIM
        PUBLIC "${PROJECT_SOURCE_DIR}"
        PUBLIC ${LIBRARY_ROOT}/VkFFT
)
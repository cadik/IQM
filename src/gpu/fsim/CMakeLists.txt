cmake_minimum_required(VERSION 3.29)
project(IQM-FSIM)

add_library(IQM-FSIM STATIC fsim.cpp
        steps/fsim_lowpass_filter.cpp
        steps/fsim_lowpass_filter.h
        steps/fsim_log_gabor.cpp
        steps/fsim_log_gabor.h
        steps/fsim_angular_filter.cpp
        steps/fsim_angular_filter.h
        steps/fsim_filter_combinations.cpp
        steps/fsim_filter_combinations.h
        steps/fsim_final_multiply.cpp
        steps/fsim_final_multiply.h
)

target_include_directories(IQM-FSIM
        PUBLIC "${PROJECT_SOURCE_DIR}"
        PUBLIC ${LIBRARY_ROOT}/VkFFT
)
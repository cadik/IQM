cmake_minimum_required(VERSION 3.29)
project(IQM LANGUAGES CXX)

# option only ever useful for debugging/visualization of intermediate results
option(ENABLE_RENDERDOC "Enable RenderDoc capture support" OFF)

# options for methods
option(SSIM "Compile SSIM Metric" ON)
option(SVD "Compile SVD Metric" ON)
option(FSIM "Compile FSIM Metric" ON)
option(FLIP "Compile FLIP Metric" ON)
option(PSNR "Compile PSNR Metric" ON)
option(LPIPS "Compile LPIPS Metric" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVULKAN_HPP_NO_CONSTRUCTORS")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(LIBRARY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(PROFILE_NAME ${PROJECT_NAME}-profile)

add_subdirectory(src)
add_subdirectory(bin)

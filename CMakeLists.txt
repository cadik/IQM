cmake_minimum_required(VERSION 3.29)
project(IQM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DVULKAN_HPP_NO_CONSTRUCTORS")

find_package( OpenCV REQUIRED )
find_package( Vulkan REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(IQM src/main.cpp
        src/methods.h
        src/args.cpp
        src/args.h
        src/methods.cpp
        src/cpu/ssim_ref.cpp
        src/cpu/ssim_ref.h
        src/gpu/base/vulkan_runtime.cpp
        src/gpu/base/vulkan_runtime.h
        src/gpu/ssim.cpp
        src/gpu/ssim.h)

target_link_libraries( IQM ${OpenCV_LIBS} Vulkan::Vulkan )

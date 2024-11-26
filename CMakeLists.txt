cmake_minimum_required(VERSION 3.29)
project(IQM)

option(ENABLE_RENDERDOC "Enable RenderDoc capture support" ON)
option(FSIM "Compile FSIM Metric" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DVULKAN_HPP_NO_CONSTRUCTORS")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(LIBRARY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib)

find_package( OpenCV REQUIRED )
find_package( Vulkan REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

if(ENABLE_RENDERDOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RENDERDOC")
endif(ENABLE_RENDERDOC)

if (FSIM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_FSIM")
    add_subdirectory(src/gpu/fsim)
endif (FSIM)

add_executable(${PROJECT_NAME} src/main.cpp
        src/methods.h
        src/args.cpp
        src/args.h
        src/methods.cpp
        src/gpu/base/vulkan_runtime.cpp
        src/gpu/base/vulkan_runtime.h
        src/gpu/ssim.cpp
        src/gpu/ssim.h
        src/debug_utils.h
        src/cpu/cw_ssim_ref.cpp
        src/cpu/cw_ssim_ref.h
        src/gpu/svd.cpp
        src/gpu/svd.h
        src/gpu/img_params.h
        src/gpu/base/vulkan_image.h
        src/timestamps.h)

add_executable(${PROJECT_NAME}-profile src/profile.cpp
        src/methods.h
        src/args.cpp
        src/args.h
        src/methods.cpp
        src/gpu/base/vulkan_runtime.cpp
        src/gpu/base/vulkan_runtime.h
        src/gpu/ssim.cpp
        src/gpu/ssim.h
        src/debug_utils.h
        src/cpu/cw_ssim_ref.cpp
        src/cpu/cw_ssim_ref.h
        src/gpu/svd.cpp
        src/gpu/svd.h
        src/gpu/img_params.h
        src/gpu/base/vulkan_image.h
        src/timestamps.h)

target_compile_definitions(${PROJECT_NAME}-profile PUBLIC PROFILE)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DVK_API_VERSION=13)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Vulkan::Vulkan )
target_compile_definitions(${PROJECT_NAME}-profile PUBLIC -DVK_API_VERSION=13)

find_package(glfw3)
target_link_libraries(${PROJECT_NAME}-profile ${OpenCV_LIBS} Vulkan::Vulkan glfw)
if (FSIM)
    target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_ROOT}/VkFFT)
    target_include_directories(${PROJECT_NAME}-profile PUBLIC ${LIBRARY_ROOT}/VkFFT)
    add_library(VkFFT INTERFACE)
    target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=0)
    target_link_libraries(${PROJECT_NAME} IQM-FSIM VkFFT SPIRV glslang)
    target_link_libraries(${PROJECT_NAME}-profile IQM-FSIM VkFFT SPIRV glslang)
endif (FSIM)
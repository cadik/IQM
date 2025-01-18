cmake_minimum_required(VERSION 3.29)
project(IQM)

# option only ever useful for debugging/visualization of intermediate results
option(ENABLE_RENDERDOC "Enable RenderDoc capture support" OFF)

# options for methods
option(SSIM "Compile SSIM Metric" ON)
option(SVD "Compile SVD Metric" ON)
option(FSIM "Compile FSIM Metric" ON)
option(FLIP "Compile FLIP Metric" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DVULKAN_HPP_NO_CONSTRUCTORS")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(LIBRARY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(PROFILE_NAME ${PROJECT_NAME}-profile)

find_package( Vulkan REQUIRED )
include_directories( ${LIBRARY_ROOT}/stb )

if(ENABLE_RENDERDOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RENDERDOC")
endif()

if (SSIM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_SSIM")
    add_subdirectory(src/gpu/ssim)
endif ()

if (SVD)
    # only used in SVD
    find_package( OpenCV REQUIRED )
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_SVD")
    add_subdirectory(src/gpu/svd)
endif (SVD)

if (FSIM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_FSIM")
    add_subdirectory(src/gpu/fsim)
endif ()

if (FLIP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_FLIP")
    add_subdirectory(src/gpu/flip)
endif ()

add_executable(${PROJECT_NAME} src/main.cpp
        src/methods.h
        src/args.cpp
        src/args.h
        src/methods.cpp
        src/gpu/base/vulkan_runtime.cpp
        src/gpu/base/vulkan_runtime.h
        src/debug_utils.h
        src/cpu/cw_ssim_ref.cpp
        src/cpu/cw_ssim_ref.h
        src/gpu/img_params.h
        src/gpu/base/vulkan_image.h
        src/timestamps.h)

add_executable(${PROFILE_NAME} src/profile.cpp
        src/methods.h
        src/args.cpp
        src/args.h
        src/methods.cpp
        src/gpu/base/vulkan_runtime.cpp
        src/gpu/base/vulkan_runtime.h
        src/debug_utils.h
        src/cpu/cw_ssim_ref.cpp
        src/cpu/cw_ssim_ref.h
        src/gpu/img_params.h
        src/gpu/base/vulkan_image.h
        src/timestamps.h
        src/input_image.h)

target_compile_definitions(${PROFILE_NAME} PUBLIC PROFILE)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DVK_API_VERSION=13)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Vulkan::Vulkan tbb)
target_compile_definitions(${PROFILE_NAME} PUBLIC -DVK_API_VERSION=13)

find_package(glfw3)
target_link_libraries(${PROJECT_NAME}-profile ${OpenCV_LIBS} Vulkan::Vulkan glfw tbb)

if (SSIM)
    target_link_libraries(${PROJECT_NAME} IQM-SSIM)
    target_link_libraries(${PROFILE_NAME} IQM-SSIM)
endif (SSIM)

if (SVD)
    target_link_libraries(${PROJECT_NAME} IQM-SVD)
    target_link_libraries(${PROFILE_NAME} IQM-SVD)

    # only used in SVD
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(${PROFILE_NAME} ${OpenCV_LIBS})
endif ()

if (FSIM)
    target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_ROOT}/VkFFT)
    target_include_directories(${PROFILE_NAME} PUBLIC ${LIBRARY_ROOT}/VkFFT)
    add_library(VkFFT INTERFACE)
    target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=0)
    target_link_libraries(${PROJECT_NAME} IQM-FSIM VkFFT SPIRV glslang)
    target_link_libraries(${PROFILE_NAME} IQM-FSIM VkFFT SPIRV glslang)
endif (FSIM)

if (FLIP)
    target_link_libraries(${PROJECT_NAME} IQM-FLIP)
    target_link_libraries(${PROFILE_NAME} IQM-FLIP)
endif (FLIP)

add_library(IQM-SSIM STATIC ssim.cpp)
add_library(IQM::SSIM ALIAS IQM-SSIM)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-SSIM
        PRIVATE "../../include"
        PRIVATE "../../shaders_out"
)

target_compile_options(IQM-SSIM PRIVATE "-Wall;-Wextra")
target_link_libraries(IQM-SSIM PUBLIC Vulkan::Vulkan IQM::LibBase)
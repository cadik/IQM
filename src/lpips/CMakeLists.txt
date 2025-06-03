add_library(IQM-LPIPS STATIC lpips.cpp)
add_library(IQM::LPIPS ALIAS IQM-LPIPS)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-LPIPS
        PRIVATE "../../include"
        PRIVATE "../../shaders_out"
)

target_compile_options(IQM-LPIPS PRIVATE "-Wall;-Wextra")
target_link_libraries(IQM-LPIPS PUBLIC Vulkan::Vulkan IQM::LibBase)
add_library(IQM-FLIP STATIC flip.cpp
        flip_color_pipeline.cpp
)

add_library(IQM::FLIP ALIAS IQM-FLIP)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-FLIP
        PRIVATE "../../include"
        PRIVATE "../../shaders_out"
)

target_compile_options(IQM-FLIP PRIVATE "-Wall;-Wextra")
target_link_libraries(IQM-FLIP PUBLIC Vulkan::Vulkan IQM::LibBase)
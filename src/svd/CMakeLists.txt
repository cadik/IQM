add_library(IQM-SVD STATIC svd.cpp)
add_library(IQM::SVD ALIAS IQM-SVD)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-SVD
        PRIVATE "../../include"
        PRIVATE "../../shaders_out"
)

target_link_libraries(IQM-SVD PUBLIC Vulkan::Vulkan IQM::LibBase)
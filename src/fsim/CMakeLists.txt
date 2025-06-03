add_library(IQM-FSIM STATIC fsim.cpp
        steps/log_gabor.cpp
        steps/angular_filter.cpp
        steps/filter_combinations.cpp
        steps/final_multiply.cpp
        steps/sum_filter_responses.cpp
        steps/noise_power.cpp
        steps/phase_congruency.cpp
        steps/estimate_energy.cpp
)
add_library(IQM::FSIM ALIAS IQM-FSIM)

find_package(Vulkan REQUIRED)

target_include_directories(IQM-FSIM
        PRIVATE "../../include"
        PRIVATE "../../shaders_out"
)

target_include_directories(IQM-FSIM SYSTEM
        PUBLIC "../../lib/VkFFT/vkFFT"
)

add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=0)
target_compile_options(IQM-FSIM PRIVATE "-Wall;-Wextra")
target_link_libraries(IQM-FSIM PUBLIC Vulkan::Vulkan IQM::LibBase SPIRV glslang)
cmake_minimum_required(VERSION 3.10)
project(n-body LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 17)

# Add the __WSL__ macro if we are compiling on WSL
if (UNIX AND NOT APPLE)
    execute_process(COMMAND uname -a OUTPUT_VARIABLE UNAME)
    if (UNAME MATCHES "WSL2")
        add_compile_definitions(__WSL__)
    endif()
endif()

add_compile_options(-O3)
include_directories(lib)

# add_library(glad third-party/src/glad.c)

# add_executable(n-body main.cu)
# target_link_libraries(n-body PRIVATE glfw GL GLEW glad)

# perf executables
add_executable(check-perf perf/tri_check.cu)
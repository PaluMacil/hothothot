set(CMAKE_CUDA_STANDARD 14)

file(GLOB lib_src "*.h" "*.cpp" "*.cu" "*.cuh")
list(REMOVE_ITEM lib_src "main.cpp")

add_library(${PROJECT_NAME}_lib STATIC ${lib_src})
add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(
        hothothot
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
set(CMAKE_CUDA_STANDARD 14)

add_library(${PROJECT_NAME}_lib STATIC Calculator.cu Calculator.cuh Configuration.cpp Configuration.h DeviceInfo.cu DeviceInfo.cuh Exceptions.cpp Exceptions.h)
add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
# (root) CMakeLists.txt
# Dan Wolf

cmake_minimum_required(VERSION 3.17)
project(hothothot LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
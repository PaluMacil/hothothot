project(tests)

add_subdirectory(lib)
message(STATUS "gtest_SOURCE_DIR is ${gtest_SOURCE_DIR}")
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

file(GLOB test_files "*.h" "*.cpp")
add_executable(${PROJECT_NAME}_run ${test_files})

target_link_libraries(${PROJECT_NAME}_run gtest gtest_main ${CMAKE_PROJECT_NAME}_lib)

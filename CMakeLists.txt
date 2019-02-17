cmake_minimum_required(VERSION 3.5.2)
project(Conv2D LANGUAGES CXX)

find_package(Armadillo 8 REQUIRED)

add_library(conv2d
    src/mesh2d.cpp
    src/flux.cpp)
target_include_directories(conv2d PUBLIC
    include)
target_include_directories(conv2d PRIVATE
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(conv2d PRIVATE
    ${ARMADILLO_LIBRARIES})

configure_file(test/test.gri test.gri COPYONLY)
add_executable(mesh2d_test
    test/mesh2d_test.cpp)
target_link_libraries(mesh2d_test PUBLIC
    conv2d)

add_executable(flux_test
    test/flux_test.cpp)
target_link_libraries(flux_test PUBLIC
    conv2d)

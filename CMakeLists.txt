cmake_minimum_required(VERSION 3.5.2)
project(Euler2dFV LANGUAGES CXX)

add_library(euler2dfv
    src/mesh2d.cpp)
target_include_directories(euler2dfv PUBLIC
    include)

configure_file(test/test.gri test.gri COPYONLY)
add_executable(mesh2d_test
    test/mesh2d_test.cpp)
target_link_libraries(mesh2d_test PUBLIC
    euler2dfv)

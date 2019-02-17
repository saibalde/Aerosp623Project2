cmake_minimum_required(VERSION 3.5.2)
project(Euler2dFV LANGUAGES CXX)

find_package(Armadillo 8 REQUIRED)

add_library(euler2dfv
    src/mesh2d.cpp)
target_include_directories(euler2dfv PUBLIC
    include)
target_include_directories(euler2dfv PRIVATE
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(euler2dfv PRIVATE
    ${ARMADILLO_LIBRARIES})

configure_file(test/test.gri test.gri COPYONLY)
add_executable(mesh2d_test
    test/mesh2d_test.cpp)
target_link_libraries(mesh2d_test PUBLIC
    euler2dfv)

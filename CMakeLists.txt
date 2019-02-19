cmake_minimum_required(VERSION 3.5.2)
project(Conv2D LANGUAGES CXX)

find_package(Armadillo 8 REQUIRED)

configure_file(test/test.gri test.gri COPYONLY)
add_executable(mesh2d_test
    src/mesh2d.cpp
    test/mesh2d_test.cpp)
target_include_directories(mesh2d_test PUBLIC
    include
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(mesh2d_test PUBLIC
    ${ARMADILLO_LIBRARIES})

add_executable(flux_test
    src/flux.cpp
    test/flux_test.cpp)
target_include_directories(flux_test PUBLIC
    include
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(flux_test PUBLIC
    ${ARMADILLO_LIBRARIES})

add_executable(conv2d
    src/mesh2d.cpp
    src/euler2d.cpp
    src/conv2d.cpp)
target_include_directories(conv2d PUBLIC
    include
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(conv2d PUBLIC
    ${ARMADILLO_LIBRARIES})

cmake_minimum_required(VERSION 3.5.2)
project(Conv2D LANGUAGES CXX)

find_package(Armadillo 8 REQUIRED)

add_subdirectory(src)

add_subdirectory(roeflux)
add_subdirectory(freestreampreserve)
add_subdirectory(firstordersolver)

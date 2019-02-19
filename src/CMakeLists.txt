add_library(conv2d
    mesh2d.cpp
    euler2d.cpp)
target_include_directories(conv2d PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(conv2d PUBLIC
    ${ARMADILLO_LIBRARIES})

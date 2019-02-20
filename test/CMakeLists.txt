configure_file(test.gri test.gri COPYONLY)

add_executable(MeshTest
    MeshTest.cpp)
target_link_libraries(MeshTest
    PUBLIC conv2d)

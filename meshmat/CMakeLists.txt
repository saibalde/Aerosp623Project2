configure_file(test.gri test.gri COPYONLY)

add_executable(meshmat
    main.cpp)
target_link_libraries(meshmat
    PUBLIC conv2d)

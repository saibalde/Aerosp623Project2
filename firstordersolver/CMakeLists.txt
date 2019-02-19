configure_file(bump0.gri bump0.gri COPYONLY)

add_executable(firstordersolver
    main.cpp)
target_link_libraries(firstordersolver
    conv2d)

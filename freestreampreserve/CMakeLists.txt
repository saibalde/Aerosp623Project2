configure_file(bump0.gri bump0.gri COPYONLY)

add_executable(freestreampreserve
    main.cpp)
target_link_libraries(freestreampreserve
    conv2d)

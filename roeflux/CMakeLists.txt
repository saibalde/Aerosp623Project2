configure_file(test.gri test.gri COPYONLY)

add_executable(roeflux
    main.cpp)
target_link_libraries(roeflux
    conv2d)

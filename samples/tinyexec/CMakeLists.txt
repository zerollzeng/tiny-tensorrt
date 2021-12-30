file(GLOB tinyexec_source
tinyexec.cpp
)
add_executable(tinyexec ${tinyexec_source})
target_compile_options(tinyexec PUBLIC -std=c++11 -Wall -Wno-deprecated -Wfloat-conversion)
target_link_libraries(tinyexec tinytrt)
target_link_libraries(tinyexec ${LIBNVINFER})
target_link_libraries(tinyexec ${LIBNVINFER_PLUGIN})
target_link_libraries(tinyexec ${LIBNVPARSERS})
target_link_libraries(tinyexec ${LIBNVONNXPARSER})
target_link_libraries(tinyexec ${ZLIB_LIBRARIES})
install(TARGETS tinyexec DESTINATION ${CMAKE_INSTALL_BINDIR})
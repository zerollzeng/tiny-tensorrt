file(GLOB sampleINT8_source
sampleINT8.cpp
)
add_executable(sampleINT8 ${sampleINT8_source})
target_compile_options(sampleINT8 PUBLIC -std=c++11 -Wall -Wno-deprecated -Wfloat-conversion)
target_link_libraries(sampleINT8 tinytrt)
target_link_libraries(sampleINT8 ${LIBNVINFER})
target_link_libraries(sampleINT8 ${LIBNVINFER_PLUGIN})
target_link_libraries(sampleINT8 ${LIBNVPARSERS})
target_link_libraries(sampleINT8 ${LIBNVONNXPARSER})
target_link_libraries(sampleINT8 ${ZLIB_LIBRARIES})
install(TARGETS sampleINT8 DESTINATION ${CMAKE_INSTALL_BINDIR})
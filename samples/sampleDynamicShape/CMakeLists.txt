file(GLOB sampleDynamicShape_source
sampleDynamicShape.cpp
)
add_executable(sampleDynamicShape ${sampleDynamicShape_source})
target_compile_options(sampleDynamicShape PUBLIC -std=c++11 -Wall -Wno-deprecated -Wfloat-conversion)
target_link_libraries(sampleDynamicShape tinytrt)
target_link_libraries(sampleDynamicShape ${LIBNVINFER})
target_link_libraries(sampleDynamicShape ${LIBNVINFER_PLUGIN})
target_link_libraries(sampleDynamicShape ${LIBNVPARSERS})
target_link_libraries(sampleDynamicShape ${LIBNVONNXPARSER})
target_link_libraries(sampleDynamicShape ${ZLIB_LIBRARIES})
install(TARGETS sampleDynamicShape DESTINATION ${CMAKE_INSTALL_BINDIR})
# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8.5)

set(TARGET_NAME inference_engine_c_api)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


include_directories(${InferenceEngine_INCLUDE_DIRS})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${SOURCE_DIR}/include/*.h)

if (BUILD_SHARED_LIBS)
    set(C_API_LIBRARY_TYPE SHARED)
else()
    set(C_API_LIBRARY_TYPE STATIC)
endif()

# create library
add_library(${TARGET_NAME} ${C_API_LIBRARY_TYPE} ${HEADERS} ${SOURCES})

if (NOT WIN32)
    target_link_libraries("${TARGET_NAME}" m)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

include_directories(${TARGET_NAME} SYSTEM PRIVATE "${SOURCE_DIR}/include")

set(install_headers ${HEADERS})
set(DLDT_C_API_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_FULL_INCLUDEDIR})

configure_file("${SOURCE_DIR}/library_config/dldt_c_api.pc.in"
    "${CMAKE_BINARY_DIR}/dldt_c_api.pc" @ONLY)

install(FILES ${install_headers} DESTINATION ${DLDT_C_API_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_BINARY_DIR}/dldt_c_api.pc" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig")
# install(TARGETS ${TARGET_NAME}
#       LIBRARY DESTINATION lib)
install(FILES "${CMAKE_BINARY_DIR}/ie_cpu_extension/libcpu_extension.so" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}")
install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})

set_target_properties("${TARGET_NAME}"
    PROPERTIES
    "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE" COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries("${TARGET_NAME}" IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} dl)

# Copyright (c) 2019 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.5)

project(InferenceEngine_C_API)

include(GNUInstallDirs)

set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

# Find InferenceEngine
find_package(InferenceEngine 1.0)
if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()

include_directories(${InferenceEngine_INCLUDE_DIRS})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

#inference_engine_c_api
set(C_API_LIB inference_engine_c_api)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file (GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        )

file (GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
        )

if (BUILD_SHARED_LIBS)
    set(C_API_LIBRARY_TYPE SHARED)
else()
    set(C_API_LIBRARY_TYPE STATIC)
endif()

add_library("${C_API_LIB}" "${C_API_LIBRARY_TYPE}" "${HEADERS}" "${SOURCES}")
if (NOT WIN32)
    target_link_libraries("${C_API_LIB}" m)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

include_directories(${C_API_LIB} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(install_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/ie_c_api.h)
set(DLDT_C_API_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_FULL_INCLUDEDIR}/dldt)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/library_config/dldt_c_api.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dldt_c_api.pc" @ONLY)

install(FILES ${install_headers} DESTINATION ${DLDT_C_API_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dldt_c_api.pc" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ie_cpu_extension/libcpu_extension.so" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}")
install(TARGETS "${C_API_LIB}" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}")

set_target_properties("${C_API_LIB}"
    PROPERTIES
    "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE" COMPILE_PDB_NAME ${C_API_LIB})

target_link_libraries("${C_API_LIB}" IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} dl)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    add_subdirectory(test)
endif()

# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8.5)
project(InferenceEngine_C_API)

include(GNUInstallDirs)

# Find InferenceEngine
find_package(InferenceEngine 1.0)
if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(src)


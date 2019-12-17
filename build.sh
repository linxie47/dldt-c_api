#!/bin/bash

# Copyright (c) 2019 Intel Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
    printf "\nINTEL_CVSDK_DIR environment variable is not set. Trying to find setupvars.sh to set it. \n"

    setvars_path=/opt/intel/openvino
    if [ -e "$setvars_path/inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
        setvars_path="$setvars_path/inference_engine/bin/setvars.sh"
    elif [ -e "$setvars_path/bin/setupvars.sh" ]; then # for OpenVINO package
        setvars_path="$setvars_path/bin/setupvars.sh"
    else
        printf "Error: setupvars.sh is not found in hardcoded paths. \n\n"
        exit 1
    fi
    if ! source $setvars_path ; then
        printf "Unable to run ./setupvars.sh. Please check its presence. \n\n"
        exit 1
    fi
fi

IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/$system_type

echo "find InferenceEngine_DIR: $InferenceEngine_DIR, IE_PLUGINS_PATH: $IE_PLUGINS_PATH"

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine. Please install it. \n\n"
    exit 1
fi

version="$(echo $INTEL_CVSDK_DIR | rev | cut -d'/' -f-1 | rev)_$(git log -1 --oneline . | head -c7)"
BASEDIR=$PWD
TEMP_DIR=~/tmp/dldt-ie-c-api
BUILD_TYPE=Release
INSTALL_DIR="/usr/local"
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE .. && make -j16 && sudo make install && \
mkdir -p $TEMP_DIR && sudo make DESTDIR=$TEMP_DIR install

# make package
tar -C $TEMP_DIR -zvcf $BASEDIR/dldt_c_api_${version}.tgz . && sudo rm -rf $TEMP_DIR && \
echo "dldt-ie_c-api-${version}.tgz created!"

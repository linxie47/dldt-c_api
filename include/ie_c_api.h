// Copyright (C) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { IE_LAYOUT_ANY = 0, IE_LAYOUT_NCHW = 1, IE_LAYOUT_NHWC = 2 } IELayout;
typedef enum { IE_PRECISION_FP32 = 10, IE_PRECISION_U8 = 40 } IEPrecision;
typedef enum {
    IE_STATUS_OK = 0,
    IE_STATUS_GENERAL_ERROR = -1,
    IE_STATUS_NOT_IMPLEMENTED = -2,
    IE_STATUS_NETWORK_NOT_LOADED = -3,
    IE_STATUS_PARAMETER_MISMATCH = -4,
    IE_STATUS_NOT_FOUND = -5,
    IE_STATUS_OUT_OF_BOUNDS = -6,
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    IE_STATUS_UNEXPECTED = -7,
    IE_STATUS_REQUEST_BUSY = -8,
    IE_STATUS_RESULT_NOT_READY = -9,
    IE_STATUS_NOT_ALLOCATED = -10,
    IE_STATUS_INFER_NOT_STARTED = -11,
    IE_STATUS_NETWORK_NOT_READ = -12
} IEStatusCode;

typedef enum {
    RAW = 0u, ///< Plain blob (default), no extra color processing required
    RGB,      ///< RGB color format
    BGR,      ///< BGR color format, default in DLDT
    RGBX,     ///< RGBX color format with X ignored during inference
    BGRX,     ///< BGRX color format with X ignored during inference
    NV12,     ///< NV12 color format represented as compound Y+UV blob
} IEColorFormat;

typedef enum { NO_RESIZE = 0, RESIZE_BILINEAR, RESIZE_AREA } IEResizeAlg;

typedef struct ie_info ie_input_info_t;
typedef struct ie_info ie_output_info_t;
typedef struct ie_network ie_network_t;
typedef struct ie_net_layer ie_net_layer_t;
typedef struct infer_request infer_request_t;
typedef struct infer_requests infer_requests_t;
typedef struct ie_plugin ie_plugin_t;
typedef struct ie_blob ie_blob_t;
typedef struct ie_blobs ie_blobs_t;
typedef struct dimensions dimensions_t;
typedef struct ie_core ie_core_t;
typedef struct roi roi_t;

/// GET IE C API VERSION ///
const char *ie_c_api_version(void);

#define MAX_DIMENSIONS 8
struct dimensions {
    size_t ranks;
    size_t dims[MAX_DIMENSIONS];
};

//// IE INPUT/OUTPUT INFO ////
struct ie_info {
    const char *name;
    dimensions_t dim;
    void *object;
};

const char *ie_info_get_name(const void *info);
const dimensions_t *ie_info_get_dimensions(const void *info);
void ie_input_info_set_precision(ie_input_info_t *info, const char *precision);
void ie_input_info_set_layout(ie_input_info_t *info, const char *layout);
void ie_input_info_set_preprocess(ie_input_info_t *info, ie_network_t *network, IEResizeAlg alg, IEColorFormat fmt);
void ie_output_info_set_precision(ie_output_info_t *info, const char *precision);

//// IE NET LAYER ////
struct ie_net_layer {};

//// IE BLOB ////

struct roi {
    size_t id;
    size_t posX;
    size_t posY;
    size_t sizeX;
    size_t sizeY;
};

struct ie_blobs {
    ie_blob_t **blobs;
    size_t num_blobs;
};

const void *ie_blob_get_data(ie_blob_t *blob);
dimensions_t *ie_blob_get_dims(ie_blob_t *blob);
IELayout ie_blob_get_layout(ie_blob_t *blob);
IEPrecision ie_blob_get_precision(ie_blob_t *blob);

//// INFER REQUEST ////
struct infer_requests {
    infer_request_t **requests;
    size_t num_reqs;
};
void infer_request_infer(infer_request_t *infer_request);
void infer_request_infer_async(infer_request_t *infer_request);
int infer_request_wait(infer_request_t *infer_request, int64_t timeout);
void *infer_request_get_blob_data(infer_request_t *infer_request, const char *name);
void infer_request_get_blob_dims(infer_request_t *infer_request, const char *name, dimensions_t *dims);
ie_blob_t *infer_request_get_blob(infer_request_t *infer_request, const char *name);
void infer_request_put_blob(ie_blob_t *blob);
int infer_request_set_blob(infer_request_t *infer_request, const char *name, size_t width, size_t height,
                           IEColorFormat format, uint8_t *data[], const roi_t *roi);

//// IE NETWORK ////
ie_network_t *ie_network_create(ie_core_t *core, const char *model, const char *weights);
void ie_network_destroy(ie_network_t *network);
size_t ie_network_get_batch_size(ie_network_t *network);
// void ie_network_add_output(ie_network_t *network, const char *out_layer, const char *precision);
// ie_net_layer_t *ie_network_get_layer(ie_network_t *network, const char *layer_name);
size_t ie_network_get_input_number(ie_network_t *network);
size_t ie_network_get_output_number(ie_network_t *network);
void ie_network_get_input(ie_network_t *network, ie_input_info_t *info, const char *input_layer_name);
void ie_network_get_output(ie_network_t *network, ie_output_info_t *info, const char *output_layer_name);
void ie_network_get_all_inputs(ie_network_t *network, ie_input_info_t **const inputs_ptr);
void ie_network_get_all_outputs(ie_network_t *network, ie_input_info_t **const outputs_ptr);
void ie_network_input_reshape(ie_network_t *network, ie_input_info_t *info, int batch_size);
/*
 * \brief Creat infer requests and return requests array
 * @return: (infer_requests *) - no memory allocation required for this value
 */
infer_requests_t *ie_network_create_infer_requests(ie_network_t *network, int num_requests, const char *device);

//// IE PLUGIN ////
ie_plugin_t *ie_plugin_create(const char *device);
void ie_plugin_destroy(ie_plugin_t *plugin);
void ie_plugin_set_config(ie_plugin_t *plugin, const char *configs);
const char *ie_plugin_get_config(ie_plugin_t *plugin, const char *config_key);
void ie_plugin_add_cpu_extension(ie_plugin_t *plugin, const char *ext_path);

//// IE CORE ////
ie_core_t *ie_core_create(void);
void ie_core_destroy(ie_core_t *core);
void ie_core_set_config(ie_core_t *core, const char *configs, const char *device);
const char *ie_core_get_config(ie_core_t *core, const char *config_key);
void ie_core_add_extension(ie_core_t *core, const char *ext_path, const char *device);

#ifdef __cplusplus
}
#endif
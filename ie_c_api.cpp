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

#include "ie_c_api.h"
#include "ie_api_impl.hpp"
#include <assert.h>

#include <ext_list.hpp>

namespace IEPY = InferenceEnginePython;
namespace IE = InferenceEngine;

#ifdef __cplusplus
extern "C" {
#endif

#define IE_C_API_VERSION_MAJOR 1
#define IE_C_API_VERSION_MINOR 1
#define IE_C_API_VERSION_PATCH 0

struct infer_request {
    void *object;
    ie_network_t *network;
};

struct ie_network {
    const char *name;
    size_t batch_size;
    void *object;
    ie_plugin_t *plugin;
    infer_requests_t *infer_requests;
    std::unique_ptr<IEPY::IEExecNetwork> ie_exec_network;
    std::map<std::string, IEPY::InputInfo> inputs;
    std::map<std::string, IEPY::OutputInfo> outputs;
};

struct ie_plugin {
    void *object;
    const char *device_name;
    const char *version;
    std::map<std::string, std::string> config;
};

struct ie_blob {
    dimensions_t dim;
    IE::Blob::Ptr object;
};

namespace {

inline std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos)
        return filepath;
    return filepath.substr(0, pos);
}

/* config string format: "A=1\nB=2\nC=3\n" */
inline std::map<std::string, std::string> String2Map(std::string const &s) {
    std::string key, val;
    std::istringstream iss(s);
    std::map<std::string, std::string> m;

    while (std::getline(std::getline(iss, key, '=') >> std::ws, val)) {
        m[key] = val;
    }

    return m;
}

const char *ie_c_api_version(void) {
    std::ostringstream ostr;
    ostr << IE_C_API_VERSION_MAJOR << "." << IE_C_API_VERSION_MINOR << "." << IE_C_API_VERSION_PATCH << std::ends;
    std::string version = ostr.str();
    return version.c_str();
}

const char *ie_info_get_name(const void *info) {
    if (info == nullptr)
        return nullptr;
    return ((const ie_info *)info)->name;
}

const dimensions_t *ie_info_get_dimensions(const void *info) {
    if (info == nullptr)
        return nullptr;
    return &((const ie_info *)info)->dim;
}

void ie_input_info_set_precision(ie_input_info_t *info, const char *precision) {
    if (info == nullptr || precision == nullptr)
        return;

    IEPY::InputInfo *info_impl = reinterpret_cast<IEPY::InputInfo *>(info->object);
    info_impl->setPrecision(precision);
}

void ie_input_info_set_layout(ie_input_info_t *info, const char *layout) {
    if (info == nullptr || layout == nullptr)
        return;

    IEPY::InputInfo *info_impl = reinterpret_cast<IEPY::InputInfo *>(info->object);
    info_impl->setLayout(layout);
}

void ie_output_info_set_precision(ie_output_info_t *info, const char *precision) {
    if (info == nullptr || precision == nullptr)
        return;

    IEPY::OutputInfo *info_impl = reinterpret_cast<IEPY::OutputInfo *>(info->object);
    info_impl->setPrecision(precision);
}

const void *ie_blob_get_data(ie_blob_t *blob) {
    return blob->object->buffer();
}

dimensions_t *ie_blob_get_dims(ie_blob_t *blob) {
    return &blob->dim;
}

IELayout ie_blob_get_layout(ie_blob_t *blob) {
    return static_cast<IELayout>((int)blob->object->getTensorDesc().getLayout());
}

IEPrecision ie_blob_get_precision(ie_blob_t *blob) {
    return static_cast<IEPrecision>((int)blob->object->getTensorDesc().getPrecision());
}

void infer_request_infer(infer_request_t *infer_request) {
    if (infer_request == nullptr)
        return;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    infer_wrap->infer();
}

void infer_request_infer_async(infer_request_t *infer_request) {
    if (infer_request == nullptr)
        return;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    infer_wrap->infer_async();
}

int infer_request_wait(infer_request_t *infer_request, int64_t timeout) {
    if (infer_request == nullptr)
        return -1;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    return infer_wrap->wait(timeout);
}

void *infer_request_get_blob_data(infer_request_t *infer_request, const char *name) {
    if (infer_request == nullptr || name == nullptr)
        return nullptr;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    InferenceEngine::Blob::Ptr blob_ptr;
    try {
        infer_wrap->getBlobPtr(name, blob_ptr);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Can not get blob: " + std::string(name)));
    }

    return blob_ptr->buffer();
}

void infer_request_get_blob_dims(infer_request_t *infer_request, const char *name, dimensions_t *dims) {
    if (infer_request == nullptr || name == nullptr || dims == nullptr)
        return;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    InferenceEngine::Blob::Ptr blob_ptr;
    try {
        infer_wrap->getBlobPtr(name, blob_ptr);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Can not get blob: " + std::string(name)));
    }

    const std::vector<size_t> ie_dims = blob_ptr->getTensorDesc().getDims();
    dims->ranks = ie_dims.size();
    for (size_t i = 0; i < dims->ranks; i++)
        dims->dims[i] = ie_dims[i];
}

ie_blob_t *infer_request_get_blob(infer_request_t *infer_request, const char *name) {
    if (infer_request == nullptr || name == nullptr)
        return nullptr;

    IEPY::InferRequestWrap *infer_wrap = reinterpret_cast<IEPY::InferRequestWrap *>(infer_request->object);
    InferenceEngine::Blob::Ptr blob_ptr;
    try {
        infer_wrap->getBlobPtr(name, blob_ptr);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Can not get blob: " + std::string(name)));
    }
    ie_blob_t *_blob = new ie_blob_t;
    assert(_blob);

    const std::vector<size_t> dims = blob_ptr->getTensorDesc().getDims();

    _blob->dim.ranks = dims.size();
    for (size_t i = 0; i < _blob->dim.ranks; i++)
        _blob->dim.dims[i] = dims[i];
    _blob->object = blob_ptr;
    return _blob;
}

void infer_request_put_blob(ie_blob_t *blob) {
    if (blob == nullptr)
        return;

    delete blob;
}

ie_network_t *ie_network_create(ie_plugin_t *plugin, const char *model, const char *weights) {
    ie_network_t *network = new ie_network_t;

    assert(plugin && model && network);
    std::string weights_file;
    if (weights == nullptr)
        weights_file = fileNameNoExt(model) + ".bin";
    else
        weights_file = weights;

    IEPY::IENetwork *ie_network_ptr = nullptr;
    try {
        ie_network_ptr = new IEPY::IENetwork(model, weights_file);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Can not create network for model: " + std::string(model)));
    }

    network->name = ie_network_ptr->name.c_str();
    network->plugin = plugin;
    network->object = ie_network_ptr;
    network->inputs = ie_network_ptr->getInputs();
    network->outputs = ie_network_ptr->getOutputs();

    return network;
}

void ie_network_destroy(ie_network_t *network) {
    if (network == nullptr)
        return;

    if (network->object)
        delete reinterpret_cast<IEPY::IENetwork *>(network->object);

    infer_requests_t *reqs = network->infer_requests;
    if (reqs) {
        for (size_t i = 0; i < reqs->num_reqs; i++)
            free(reqs->requests[i]);
        free(reqs->requests);
        free(reqs);
    }

    delete network;
}

void ie_network_set_batch(ie_network_t *network, const size_t size) {
    if (network == nullptr)
        return;

    IEPY::IENetwork *network_impl = reinterpret_cast<IEPY::IENetwork *>(network->object);
    network_impl->setBatch(size);
    network->batch_size = size;
}

size_t ie_network_get_batch_size(ie_network_t *network) {
    if (network == nullptr)
        return 0;

    IEPY::IENetwork *network_impl = reinterpret_cast<IEPY::IENetwork *>(network->object);
    return network_impl->actual.getBatchSize();
}

void ie_network_add_output(ie_network_t *network, const char *out_layer, const char *precision) {
    // TODO
}

size_t ie_network_get_input_number(ie_network_t *network) {
    if (network == nullptr)
        return 0;

    return network->inputs.size();
}

size_t ie_network_get_output_number(ie_network_t *network) {
    if (network == nullptr)
        return 0;

    return network->outputs.size();
}

static inline void ConvertToInputInfo(std::map<std::string, IEPY::InputInfo>::iterator it, ie_info *info) {
    info->name = it->first.c_str();
    IEPY::InputInfo *input_info = &it->second;
    info->dim.ranks = input_info->dims.size();
    for (size_t i = 0; i < info->dim.ranks; i++)
        info->dim.dims[i] = input_info->dims[i];
    info->object = input_info;
}

static inline void ConvertToOutputInfo(std::map<std::string, IEPY::OutputInfo>::iterator it, ie_info *info) {
    info->name = it->first.c_str();
    IEPY::OutputInfo *output_info = &it->second;
    info->dim.ranks = output_info->dims.size();
    for (size_t i = 0; i < info->dim.ranks; i++)
        info->dim.dims[i] = output_info->dims[i];
    info->object = output_info;
}

void ie_network_get_input(ie_network_t *network, ie_input_info_t *info, const char *input_layer_name) {
    if (network == nullptr || info == nullptr)
        return;

    auto it = (input_layer_name == nullptr) ? network->inputs.begin() : network->inputs.find(input_layer_name);
    if (it != network->inputs.end()) {
        ConvertToInputInfo(it, info);
    }
}

void ie_network_get_output(ie_network_t *network, ie_output_info_t *info, const char *output_layer_name) {
    if (network == nullptr || info == nullptr)
        return;

    auto it = (output_layer_name == nullptr) ? network->outputs.begin() : network->outputs.find(output_layer_name);
    if (it != network->outputs.end()) {
        ConvertToOutputInfo(it, info);
    }
}

void ie_network_get_all_inputs(ie_network_t *network, ie_input_info_t **const inputs_ptr) {
    if (network == nullptr || inputs_ptr == nullptr)
        return;

    size_t index = 0;
    for (auto it = network->inputs.begin(); it != network->inputs.end(); it++) {
        assert(inputs_ptr[index]);
        ConvertToInputInfo(it, inputs_ptr[index]);
        index++;
    }
}

void ie_network_get_all_outputs(ie_network_t *network, ie_input_info_t **const outputs_ptr) {
    if (network == nullptr || outputs_ptr == nullptr)
        return;

    size_t index = 0;
    for (auto it = network->outputs.begin(); it != network->outputs.end(); it++) {
        assert(outputs_ptr[index]);
        ConvertToOutputInfo(it, outputs_ptr[index]);
        index++;
    }
}

infer_requests_t *ie_network_create_infer_requests(ie_network_t *network, int num_requests) {
    assert(network && network->plugin && num_requests > 0);

    IEPY::IEPlugin *plugin = reinterpret_cast<IEPY::IEPlugin *>(network->plugin->object);
    try {
        network->ie_exec_network =
            plugin->load(*reinterpret_cast<IEPY::IENetwork *>(network->object), num_requests, network->plugin->config);
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Failed to load network!"));
    }

    infer_requests_t *requests = (decltype(requests))malloc(sizeof(*requests));
    infer_request_t **request_ptrs = (decltype(request_ptrs))malloc(num_requests * sizeof(infer_request_t *));

    assert(requests && request_ptrs);

    requests->num_reqs = num_requests;
    for (int n = 0; n < num_requests; n++) {
        request_ptrs[n] = (infer_request *)malloc(sizeof(infer_request_t));
        assert(request_ptrs[n]);
        request_ptrs[n]->object = &network->ie_exec_network->infer_requests[n];
        request_ptrs[n]->network = network;
    }
    requests->requests = request_ptrs;

    network->infer_requests = requests;

    return requests;
}

ie_plugin_t *ie_plugin_create(const char *device) {
    ie_plugin_t *plugin = new ie_plugin_t;

    assert(device && plugin);

    IEPY::IEPlugin *ie_plugin_ptr = nullptr;
    std::string plugin_dir("");
    try {
        ie_plugin_ptr = new IEPY::IEPlugin(device, {plugin_dir});
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Can not create plugin for device: " + std::string(device)));
    }

    plugin->device_name = ie_plugin_ptr->device_name.c_str();
    plugin->version = ie_plugin_ptr->version.c_str();
    plugin->object = ie_plugin_ptr;

    std::cout << "Devivce:" << ie_plugin_ptr->device_name << " Ver:" << ie_plugin_ptr->version << std::endl;

    return plugin;
}

void ie_plugin_destroy(ie_plugin_t *plugin) {
    if (plugin == nullptr)
        return;

    if (plugin->object)
        delete reinterpret_cast<IEPY::IEPlugin *>(plugin->object);

    delete plugin;
}

void ie_plugin_set_config(ie_plugin *plugin, const char *ie_configs) {
    if (plugin == nullptr || ie_configs == nullptr)
        return;

    IEPY::IEPlugin *plugin_impl = reinterpret_cast<IEPY::IEPlugin *>(plugin->object);
    plugin->config = String2Map(ie_configs);
    // TODO: Inference Engine asserts if unknown key passed
    std::map<std::string, std::string> ie_config(plugin->config);
    ie_config.erase("RESIZE_BY_INFERENCE");
    ie_config.erase("CPU_EXTENSION");
    plugin_impl->setConfig(ie_config);
};

const char *ie_plugin_get_config(ie_plugin_t *plugin, const char *config_key) {
    if (plugin == nullptr || config_key == nullptr)
        return nullptr;

    auto it = plugin->config.find(config_key);
    if (it != plugin->config.end())
        return it->second.c_str();

    return nullptr;
}

void ie_plugin_add_cpu_extension(ie_plugin_t *plugin, const char *ext_path) {
    if (plugin == nullptr)
        return;

    IEPY::IEPlugin *plugin_impl = reinterpret_cast<IEPY::IEPlugin *>(plugin->object);

    if (ext_path == nullptr) {
        InferenceEngine::ResponseDesc response;
        plugin_impl->actual->AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(),
                                          &response);
    } else {
        plugin_impl->addCpuExtension(ext_path);
    }
}

} // namespace

#ifdef __cplusplus
}
#endif

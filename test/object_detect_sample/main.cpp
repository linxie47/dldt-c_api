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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include "ie_c_api.h"
#include "test.h"

const char *CONFIGS = "CPU_THREADS_NUM=4|CPU_THROUGHPUT_STREAMS=4|CPU_BIND_THREAD=NO";

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
// ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat& frame, infer_request_t *infer_request,
                 const char *inputName, int batch_id)
{
    dimensions_t dimenison;
    infer_request_get_blob_dims(infer_request, inputName, &dimenison);

    size_t channels = dimenison.dims[1];
    size_t width = dimenison.dims[3];
    size_t height = dimenison.dims[2];
    uint8_t *blob_data = (uint8_t *)infer_request_get_blob_data(infer_request, inputName);
    cv::Mat resized_image;
    cv::resize(frame, resized_image, cv::Size(300, 300));

    int batchoffset = batch_id * channels * width * height;

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[c * width * height + h * width + w + batchoffset] =
                        resized_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

int main(int argc, char *argv[])
{
// --------------------------- 1. Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }
// --------------------------- 2. Read input -----------------------------------------------------------
    std::cout << "Loading a Picture" << std::endl;
    cv::VideoCapture cap;
    if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
        throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
    }
    const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // read input (video) frame
    cv::Mat curr_frame;  cap >> curr_frame;
    cv::Mat next_frame;

    if (!cap.grab()) {
        throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                               "Failed getting next frame from the " + FLAGS_i);
    }

    int batch_size = FLAGS_b;
    std::cout << "Batch size is " << std::to_string(batch_size) << std::endl;

// --------------------------- 3. Load Plugin for inference engine -------------------------------------
    std::cout << "Loading Core" << std::endl;
    const char *device = FLAGS_d.data();
    if (device == nullptr)
        device = "CPU";

    ie_core_t *core = ie_core_create();
    ie_core_set_config(core, CONFIGS, device);
    ie_core_add_extension(core, NULL, device);

// --------------------------- 4. Read IR (.xml and .bin files) and Loading model to the plugin ------------
    const char *model = FLAGS_m.data();
    const char *weight = nullptr;
    std::cout << "Loading Network" << std::endl;

    ie_network_t *ie_network = ie_network_create(core, model, weight);
// --------------------------- 5. Prepare input blobs --------------------------------------------------
    std::cout << "Preparing input blobs" << std::endl;

    /** SSD network has one input and one output **/
    const char *input_name = nullptr;
    ie_input_info_t input_info;

    ie_network_get_input(ie_network, &input_info, input_name);

    if (batch_size > 1)
        ie_network_input_reshape(ie_network, &input_info, batch_size);

    const char *inputPrecision = "U8";
    ie_input_info_set_precision(&input_info, inputPrecision);
    const char *Layout = "NCHW";
    ie_input_info_set_layout(&input_info, Layout);

// --------------------------- 6. Prepare output blobs -------------------------------------------------
    std::cout << "Preparing output blobs" << std::endl;
    const char *output_name = nullptr;
    ie_output_info_t output_info;

    ie_network_get_output(ie_network, &output_info, output_name);

    const int maxProposalCount = output_info.dim.dims[2];
    const int objectSize = output_info.dim.dims[3];

    if (objectSize != 7) {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }
    const char *outputPrecision = "FP32";
    ie_output_info_set_precision(&output_info, outputPrecision);
// --------------------------- 7. Create infer request -------------------------------------------------
    int num_requests = FLAGS_nireq;
    infer_requests_t *infer = ie_network_create_infer_requests(ie_network, num_requests, device);
    std::queue<infer_request_t *> available_infer, pending_infer;
    for (int i = 0; i < num_requests; i++) {
        available_infer.push(infer->requests[i]);
    }

// --------------------------- 8. Do inference --------------------------------------------------------
    bool isLastFrame = false;

    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto total_t0 = std::chrono::high_resolution_clock::now();

    while (true) {
        // Here is the first asynchronous point:
        // in the async mode we capture frame to populate the NEXT infer request
        // in the regular mode we capture frame to the CURRENT infer request
        if (!cap.read(next_frame)) {
            if (next_frame.empty()) {
                isLastFrame = true;  // end of video file
            } else {
                throw std::logic_error("Failed to get frame from cv::VideoCapture");
            }
        }

        if (available_infer.empty()) {
            infer_request_t *async_infer_request = pending_infer.front();

            if (0 == infer_request_wait(async_infer_request, -1)) {
                const float* detection = (const float *)infer_request_get_blob_data(async_infer_request, output_info.name);

                /* Each detection has image_id that denotes processed image */
                for (int i = 0; i < maxProposalCount; i++) {
                    auto image_id = static_cast<int>(detection[i * objectSize + 0]);
                    if (image_id < 0) {
                        break;
                    }

                    float confidence = detection[i * objectSize + 2];
                    auto label = static_cast<int>(detection[i * objectSize + 1]);
                    float xmin = static_cast<int>(detection[i * objectSize + 3] * width);
                    float ymin = static_cast<int>(detection[i * objectSize + 4] * height);
                    float xmax = static_cast<int>(detection[i * objectSize + 5] * width);
                    float ymax = static_cast<int>(detection[i * objectSize + 6] * height);

                    if (confidence > 0.5)
                        std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
                            "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id << std::endl;
                }
            }

            pending_infer.pop();
            available_infer.push(async_infer_request);
        }

        infer_request_t *async_infer_request = available_infer.front();

        if (!isLastFrame) {
            for (int b = 0; b < batch_size; b++) {
                frameToBlob(next_frame, async_infer_request, input_info.name, b);
            }
        }

        if (!isLastFrame) {
            infer_request_infer_async(async_infer_request);
        }
        available_infer.pop();
        pending_infer.push(async_infer_request);

        if (isLastFrame) {
            break;
        }
    }
    /** Show performace results **/
    auto total_t1 = std::chrono::high_resolution_clock::now();
    ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
    std::cout << "Total Inference time: " << total.count() << std::endl;

    ie_network_destroy(ie_network);
    ie_core_destroy(core);
    return 0;
}

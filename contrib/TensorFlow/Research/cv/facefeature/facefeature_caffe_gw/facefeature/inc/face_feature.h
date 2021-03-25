/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File sample_process.h
* Description: handle acl resource
*/
#pragma once
#include "utils.h"
#include "acl/acl.h"
#include <memory>
#include "model_process.h"

using namespace std;

/**
* ClassifyProcess
*/
class FaceFeature {
public:
    FaceFeature(const char* modelPath,
        uint32_t modelWidth, uint32_t modelHeight);
    ~FaceFeature();

    Result init();
    Result pre_process(const string&, ImageData& srcImage);
    Result inference(aclmdlDataset*& inferenceOutput, ImageData& resizedImage);
    Result post_process(aclmdlDataset* modelOutput,FaceInfo& faceInfo);
private:
    Result init_resource();
    Result init_model(const char* omModelPath);
    Result create_image_info_buffer();
    void* get_inference_output_item(uint32_t& itemDataSize,
                                 aclmdlDataset* inferenceOutput,
                                 uint32_t idx);
    void destroy_resource();

private:
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    uint32_t imageInfoSize_;
    void* imageInfoBuf_;
    ModelProcess model_;

    const char* modelPath_;
    uint32_t inputDataSize_;

    aclrtRunMode runMode_;

    bool isInited_;

    uint32_t imageCount = 0;

    uint32_t modelWidth_;
    uint32_t modelHeight_;
};


/* *
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _ASYN_E2E_INFERENCE_ENGINE_H_
#define _ASYN_E2E_INFERENCE_ENGINE_H_

#include "inference_engine.h"

class Asyn_e2e_inference_engine : public Asyn_InferEngine {
public:
    uint32_t inputSize_;
    void *inputAddr_;
    std::vector<ImgInfo> imgInfo_vec_;

public:
    Asyn_e2e_inference_engine();
    Asyn_e2e_inference_engine(Config *config);
    Asyn_e2e_inference_engine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue);
    ~Asyn_e2e_inference_engine();

    virtual aclError InferenceThreadProc();
    virtual aclError Init(Config *config);
    virtual aclError CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec);
    virtual aclError ExecInference();
};
#endif

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
#ifndef _E2E_INFERENCE_ENGINE_H_
#define _E2E_INFERENCE_ENGINE_H_

#include "inference_engine.h"


class E2e_Inference_engine : public InferEngine {
public:
    uint32_t inputSize_;
    void *inputAddr_;
    std::vector<ImgInfo> imgInfo_vec_;
    DetBox *detBox_;

public:
    E2e_Inference_engine();
    E2e_Inference_engine(Config *config);
    E2e_Inference_engine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue);
    ~E2e_Inference_engine();

    virtual aclError InferenceThreadProc();
    virtual aclError Init(Config *config);
    virtual aclError CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec);
    virtual aclError CreateInferInput(std::vector<std::string> &inferFile_vec);
    virtual aclError ParseDetInferResult(std::vector<std::string> &inferFile_vec);
    virtual aclError AddDetRectInImg(std::vector<std::string> &inferFile_vec);
    virtual aclError DetInferPost(std::vector<std::string> &inferFile_vec);
    virtual aclError TearDown();
};


#endif

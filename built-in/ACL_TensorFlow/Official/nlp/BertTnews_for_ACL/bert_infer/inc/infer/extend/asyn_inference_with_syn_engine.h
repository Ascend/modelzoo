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

#ifndef _ASYN_INFERENCE_WITH_SYN_ENGINE_H
#define _ASYN_INFERENCE_WITH_SYN_ENGINE_H

#include "inference_engine.h"

class Asyn_Inference_with_syn_engine : public Asyn_InferEngine {
public:
    uint32_t curDevIndex_;
    uint32_t curChnIndex_;
    uint8_t synStreamFinish_;

public:
    Asyn_Inference_with_syn_engine();
    Asyn_Inference_with_syn_engine(Config *config);
    Asyn_Inference_with_syn_engine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue);
    ~Asyn_Inference_with_syn_engine();

    virtual aclError InferenceThreadProc();

    aclError ExecInference();

    virtual void UnSubscribeAndDtyStream();

private:
};


#endif

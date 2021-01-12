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

#include "asyn_inference_with_syn_engine.h"
class Asyn_Waitevent_InferEngine : public Asyn_InferEngine {
public:
    uint32_t threadindex;
    pthread_t pid;
    uint32_t threadnum;

public:
    Asyn_Waitevent_InferEngine();
    Asyn_Waitevent_InferEngine(Config *config);
    Asyn_Waitevent_InferEngine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
        BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
        : Asyn_InferEngine(inQue, outQue)
    {}
    ~Asyn_Waitevent_InferEngine();

    aclError ExecInference_w(aclrtStream inferStream);
    aclError InferenceThreadProc_w();

private:
};
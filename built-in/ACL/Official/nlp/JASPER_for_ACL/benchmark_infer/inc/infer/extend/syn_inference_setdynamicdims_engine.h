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

#include "inference_engine.h"

class Syn_Inference_SetDynamicDims_engine : public InferEngine {
public:
    Syn_Inference_SetDynamicDims_engine();
    Syn_Inference_SetDynamicDims_engine(Config *config);
    ~Syn_Inference_SetDynamicDims_engine();

    virtual aclError InferenceThreadProc();

    aclError ExecInference(int gear_count);

    aclError SetDynamicDims(int gear_count);

    aclError GetDynamicBatch();

    aclError GetDynamicHW();

    aclError GetDynamicGearCount();

    aclError InferenceThread_GetDynamicBatch();

    aclError InferenceThread_GetDynamicHW();

    aclError InferenceThread_GetDynamicGearCount();
};

aclError InferDeviceContexInit(std::vector<uint32_t> &device_vec, std::vector<aclrtContext> &contex_vec);
aclError InferDestoryRsc(std::vector<uint32_t> device_vec, std::vector<aclrtContext> contex_vec);

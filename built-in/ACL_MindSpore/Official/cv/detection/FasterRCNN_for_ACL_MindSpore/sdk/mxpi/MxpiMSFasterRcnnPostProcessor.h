/*
 * Copyright(C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MXPLUGINS_FASTERRCNNPOSTPROCESSOR_H
#define MXPLUGINS_FASTERRCNNPOSTPROCESSOR_H

#include "MSFasterRcnnPostProcess.h"
#include "MxPlugins/ModelPostProcessors/ModelPostProcessorBase/MxpiObjectPostProcessorBase.h"

class MxpiMSFasterRcnnPostProcessor : public MxPlugins::MxpiObjectPostProcessorBase {
public:
    APP_ERROR Init(const std::string &configPath, const std::string &labelPath, MxBase::ModelDesc modelDesc);
    APP_ERROR DeInit() override;
    APP_ERROR Process(std::shared_ptr<void> &metaDataPtr,
                      MxBase::PostProcessorImageInfo postProcessorImageInfo,
                      std::vector<MxTools::MxpiMetaHeader> &headerVec,
                      std::vector<std::vector<MxBase::BaseTensor>> &tensors) override;

private:
    MxBase::MSFasterRcnnPostProcessor postProcessorInstance_;
};

extern "C" {
std::shared_ptr<MxPlugins::MxpiModelPostProcessorBase> GetInstance();
}

#endif  // MXPLUGINS_FASTERRCNNPOSTPROCESSOR_H

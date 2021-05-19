/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef CTPN_CLASSIFY_H
#define CTPN_CLASSIFY_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "TextObjectPostProcessors/CtpnPostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"

struct InitParam {
    uint32_t deviceId;
    uint32_t maxHorizontalGap;
    float boxIouThresh;
    float TextIouThresh;
    float TextProposalsMinScore;
    float LineMinScore;
    bool IsOriented;
    bool IsMindspore;
    bool checkModelFlag;
    std::string labelPath;
    std::string modelPath;
};

class CtpnDetection {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor, int& oriWidth, int& oriHeight);
    APP_ERROR Resize(const MxBase::TensorBase &input, MxBase::TensorBase &output);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase>& tensors,
            std::vector<std::vector<MxBase::TextObjectInfo>> &textObjInfos,
            const std::vector<MxBase::ResizedImageInfo> &resizedImageInfo);
    APP_ERROR Process(const std::string &imgPath);

private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::CrnnPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    MxBase::ResizedImageInfo resizedImageInfo_;
    uint32_t deviceId_ = 0;
};
#endif
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

#ifndef INFER_MSMASKRCNNPOSTPROCESS_H
#define INFER_MSMASKRCNNPOSTPROCESS_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"

namespace {
const int DEFAULT_CLASS_NUM_MS_MASK = 80;
const float DEFAULT_SCORE_THRESH_MS_MASK = 0.7;
const float DEFAULT_IOU_THRESH_MS_MASK = 0.5;
const int DEFAULT_RPN_MAX_NUM_MS_MASK = 1000;
const int DEFAULT_MAX_PER_IMG_MS_MASK = 128;
const float DEFAULT_THR_BINARY_MASK = 0.5;
const int DEFAULT_MASK_SIZE_MS_MASK = 28;
const std::string PREDICT_RESULT_PATH = "predict_result.json";
} // namespace

namespace MxBase {

class MSMaskRcnnPostProcessor : public MxBase::ObjectPostProcessorBase {
public:
    APP_ERROR Init(const std::string& configPath, const std::string& labelPath, MxBase::ModelDesc modelDesc) override;
    APP_ERROR Process(
        std::vector<std::shared_ptr<void>>& featLayerData,
        std::vector<ObjDetectInfo>& objInfos,
        const bool useMpPictureCrop,
        MxBase::PostImageInfo postImageInfo) override;
    APP_ERROR DeInit() override;

    static void FreeMaskMemory(std::vector<ObjDetectInfo>& objInfos);

private:
    APP_ERROR CheckMSModelCompatibility();

    void ObjectDetectionOutput(
        std::vector<std::shared_ptr<void>>& featLayerData,
        std::vector<ObjDetectInfo>& objInfos,
        ImageInfo& imgInfo) override;

    void GetValidDetBoxes(
        std::vector<std::shared_ptr<void>>& featLayerData,
        std::vector<MxBase::DetectBox>& detBoxes,
        ImageInfo& imgInfo) const;

    void ConvertObjInfoFromDetectBox(
        std::vector<MxBase::DetectBox>& detBoxes,
        std::vector<ObjDetectInfo>& objInfos,
        ImageInfo& imgInfo);

    APP_ERROR MaskPostProcess(ObjDetectInfo& objInfo, void* maskPtr, ImageInfo& imgInfo);

private:
    int classNum_ = DEFAULT_CLASS_NUM_MS_MASK;
    float scoreThresh_ = DEFAULT_SCORE_THRESH_MS_MASK;
    float iouThresh_ = DEFAULT_IOU_THRESH_MS_MASK;
    int rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS_MASK;
    int maxPerImg_ = DEFAULT_MAX_PER_IMG_MS_MASK;
    float maskThrBinary_ = DEFAULT_THR_BINARY_MASK;
    int maskSize_ = DEFAULT_MASK_SIZE_MS_MASK;
    bool saveResultToJson_ = true;
};

} // namespace MxBase
#endif // INFER_MSMASKRCNNPOSTPROCESS_H

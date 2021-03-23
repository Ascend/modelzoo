/*
 * Copyright(C) 2020 Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef INFER_MSFASTERRCNNPOSTPROCESS_H
#define INFER_MSFASTERRCNNPOSTPROCESS_H

#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"

namespace {
const float DEFAULT_SCORE_THRESH_MS_FASTER = 0.7;
const int DEFAULT_CLASS_NUM_MS_FASTER = 80;
const float DEFAULT_IOU_THRESH_MS_FASTER = 0.5;
const int DEFAULT_MAX_PER_IMG_MS_FASTER = 128;
const int DEFAULT_RPN_MAX_NUM_MS_FASTER = 1000;
}  // namespace

namespace MxBase {

class MSFasterRcnnPostProcessor : public MxBase::ObjectPostProcessorBase {
public:
    APP_ERROR
    Init(const std::string &configPath, const std::string &labelPath, MxBase::ModelDesc modelDesc) override;

    /*
     * @description: Do nothing temporarily.
     * @return APP_ERROR error code.
     */
    APP_ERROR DeInit() override;

    /*
     * @description: Get the info of detected object from output and resize to
     * original coordinates.
     * @param featLayerData Vector of output feature data.
     * @param objInfos Address of output object infos.
     * @param useMpPictureCrop if true, offsets of coordinates will be given.
     * @param postImageInfo Info of model/image width and height, offsets of
     * coordinates.
     * @return: ErrorCode.
     */
    APP_ERROR Process(std::vector<std::shared_ptr<void>> &featLayerData, std::vector<ObjDetectInfo> &objInfos,
                      const bool useMpPictureCrop, MxBase::PostImageInfo postImageInfo) override;

private:
    APP_ERROR CheckMSModelCompatibility();

    void ObjectDetectionOutput(std::vector<std::shared_ptr<void>> &featLayerData, std::vector<ObjDetectInfo> &objInfos,
                               ImageInfo &imgInfo) override;

    void GetValidDetBoxes(std::vector<std::shared_ptr<void>> &featLayerData, std::vector<MxBase::DetectBox> &detBoxes,
                          ImageInfo &imgInfo) const;

    void ConvertObjInfoFromDetectBox(std::vector<MxBase::DetectBox> &detBoxes, std::vector<ObjDetectInfo> &objInfos,
                                     ImageInfo &imgInfo) const;

    APP_ERROR ReadConfigParams();

private:
    float scoreThresh_ = DEFAULT_SCORE_THRESH_MS_FASTER;
    int classNum_ = DEFAULT_CLASS_NUM_MS_FASTER;
    float iouThresh_ = DEFAULT_IOU_THRESH_MS_FASTER;
    int maxPerImg_ = DEFAULT_MAX_PER_IMG_MS_FASTER;
    int rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS_FASTER;
};

}  // namespace MxBase

#endif  // INFER_MSFASTERRCNNPOSTPROCESS_H

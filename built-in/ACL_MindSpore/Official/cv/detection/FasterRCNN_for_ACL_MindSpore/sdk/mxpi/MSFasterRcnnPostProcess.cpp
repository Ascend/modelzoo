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

#include "MSFasterRcnnPostProcess.h"
#include "acl/acl.h"

namespace {
// Output Tensor
const int OUTPUT_TENSOR_SIZE = 3;

const int OUTPUT_BBOX_INDEX = 0;
const int OUTPUT_CLASS_INDEX = 1;
const int OUTPUT_MASK_INDEX = 2;

} // namespace

namespace {
const int LEFTTOPY = 0;
const int LEFTTOPX = 1;
const int RIGHTBOTY = 2;
const int RIGHTBOTX = 3;
} // namespace

namespace MxBase {
APP_ERROR MSFasterRcnnPostProcessor::CheckMSModelCompatibility() {
    if (outputTensorShapes_.size() != OUTPUT_TENSOR_SIZE) {
        LogError << "The size of output tensor is wrong, output size(" << outputTensorShapes_.size() << ")";
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    int total_num = classNum_ * rpnMaxNum_;

    if (outputTensorShapes_[OUTPUT_BBOX_INDEX].size() != 3) {
        LogError << "outputTensorShapes_[1].size(): " << outputTensorShapes_[OUTPUT_BBOX_INDEX].size();
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_BBOX_INDEX][1] != total_num) {
        LogError << "The output tensor is mismatched: (" << total_num << "/"
                 << outputTensorShapes_[OUTPUT_BBOX_INDEX][1] << ")"
                 << "Please check that the hyper-parameter(classNum_, rpnMaxNum_) are configured correctly.";
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_BBOX_INDEX][2] != 5) {
        LogError << "outputTensorShapes_[OUTPUT_BBOX_INDEX][2]: " << outputTensorShapes_[OUTPUT_BBOX_INDEX][2];
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_CLASS_INDEX][1] != total_num) {
        LogError << "outputTensorShapes_[OUTPUT_CLASS_INDEX][1]: " << outputTensorShapes_[OUTPUT_CLASS_INDEX][1];
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_MASK_INDEX][1] != total_num) {
        LogError << "outputTensorShapes_[OUTPUT_MASK_INDEX][1]: " << outputTensorShapes_[OUTPUT_MASK_INDEX][1];
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    return APP_ERR_OK;
}

/*
 * @description Load the configs and labels from the file.
 * @param labelPath config path and label path.
 * @return APP_ERROR error code.
 */
APP_ERROR MSFasterRcnnPostProcessor::Init(
    const std::string& configPath,
    const std::string& labelPath,
    MxBase::ModelDesc modelDesc) {
    LogInfo << "Begin to initialize MSFasterRcnnPostProcessor.";
    APP_ERROR ret = LoadConfigDataAndLabelMap(configPath, labelPath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to LoadConfigDataAndLabelMap in MSFasterRcnnPostProcessor.";
        return ret;
    }

    ret = configData_.GetFileValue<int>("CLASS_NUM", classNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No CLASS_NUM in config file, default value(" << classNum_ << ").";
    }

    ret = configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No SCORE_THRESH in config file, default value(" << scoreThresh_ << ").";
    }

    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No IOU_THRESH in config file, default value(" << iouThresh_ << ").";
    }

    ret = configData_.GetFileValue<int>("MAX_PER_IMG", maxPerImg_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MAX_PER_IMG in config file, default value(" << maxPerImg_ << ").";
    }

    ret = configData_.GetFileValue<int>("RPN_MAX_NUM", rpnMaxNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No RPN_MAX_NUM in config file, default value(" << rpnMaxNum_ << ").";
    }

    LogInfo << "The hyper-parameter are as follows: \n"
            << "    CLASS_NUM: " << classNum_ << " \n"
            << "    SCORE_THRESH: " << scoreThresh_ << " \n"
            << "    IOU_THRESH: " << iouThresh_ << " \n"
            << "    RPN_MAX_NUM: " << rpnMaxNum_ << " \n"
            << "    MAX_PER_IMG: " << maxPerImg_;

    GetModelTensorsShape(modelDesc);
    if (checkModelFlag_) {
        ret = CheckMSModelCompatibility();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to CheckModelCompatibility in MSFasterRcnnPostProcessor."
                     << "Please check the compatibility between model and postprocessor";
            return ret;
        }
    } else {
        LogWarn << "Compatibility check for model is skipped as CHECK_MODEL is set as false, please ensure your model"
                << "is correct before running.";
    }

    LogInfo << "End to initialize MSFasterRcnnPostProcessor.";
    return APP_ERR_OK;
}

/*
 * @description: Do nothing temporarily.
 * @return: APP_ERROR error code.
 */
APP_ERROR MSFasterRcnnPostProcessor::DeInit() {
    LogInfo << "Begin to deinitialize MSFasterRcnnPostProcessor.";
    LogInfo << "End to deinitialize MSFasterRcnnPostProcessor.";
    return APP_ERR_OK;
}

APP_ERROR MSFasterRcnnPostProcessor::Process(
    std::vector<std::shared_ptr<void>>& featLayerData,
    std::vector<ObjDetectInfo>& objInfos,
    const bool useMpPictureCrop,
    MxBase::PostImageInfo postImageInfo) {
    LogDebug << "Begin to process MSFasterRcnnPostProcessor.";
    ObjectPostProcessorBase::Process(featLayerData, objInfos, useMpPictureCrop, postImageInfo);
    LogDebug << "End to process MSFasterRcnnPostProcessor.";
    return APP_ERR_OK;
}

static bool CompareDetectBoxes(const MxBase::DetectBox& box1, const MxBase::DetectBox& box2) {
    return box1.prob > box2.prob;
}

static void GetDetectBoxesTopK(std::vector<MxBase::DetectBox>& detBoxes, size_t kVal) {
    std::sort(detBoxes.begin(), detBoxes.end(), CompareDetectBoxes);
    if (detBoxes.size() <= kVal) {
        return;
    }

    LogDebug << "TopK: total detect boxes: " << detBoxes.size() << ", kVal: " << kVal;
    detBoxes.erase(detBoxes.begin() + kVal, detBoxes.end());
}

void MSFasterRcnnPostProcessor::GetValidDetBoxes(
    std::vector<std::shared_ptr<void>>& featLayerData,
    std::vector<MxBase::DetectBox>& detBoxes,
    ImageInfo& imgInfo) const {
    auto* bboxPtr = static_cast<aclFloat16*>(featLayerData[OUTPUT_BBOX_INDEX].get()); // 1 * 80000 * 5
    auto* labelPtr = static_cast<int32_t*>(featLayerData[OUTPUT_CLASS_INDEX].get()); // 1 * 80000 * 1
    auto* maskPtr = static_cast<bool*>(featLayerData[OUTPUT_MASK_INDEX].get()); // 1 * 80000 * 1
    // mask filter
    float prob = 0;
    size_t total = rpnMaxNum_ * classNum_;
    for (size_t index = 0; index < total; ++index) {
        if (!maskPtr[index]) {
            continue;
        }
        size_t startIndex = index * 5;
        prob = aclFloat16ToFloat(bboxPtr[startIndex + 4]);
        if (prob <= scoreThresh_) {
            continue;
        }

        MxBase::DetectBox detBox;
        float x1 = aclFloat16ToFloat(bboxPtr[startIndex + 0]);
        float y1 = aclFloat16ToFloat(bboxPtr[startIndex + 1]);
        float x2 = aclFloat16ToFloat(bboxPtr[startIndex + 2]);
        float y2 = aclFloat16ToFloat(bboxPtr[startIndex + 3]);
        detBox.x = (x1 + x2) / COORDINATE_PARAM;
        detBox.y = (y1 + y2) / COORDINATE_PARAM;
        detBox.width = x2 - x1;
        detBox.height = y2 - y1;
        detBox.prob = prob;
        detBox.classID = labelPtr[index];

        detBoxes.push_back(detBox);
    }
    GetDetectBoxesTopK(detBoxes, maxPerImg_);
}

void MSFasterRcnnPostProcessor::ConvertObjInfoFromDetectBox(
    std::vector<MxBase::DetectBox>& detBoxes,
    std::vector<ObjDetectInfo>& objInfos,
    ImageInfo& imgInfo) const {
    float widthScale = (float)imgInfo.imgWidth / (float)imgInfo.modelWidth;
    float heightScale = (float)imgInfo.imgHeight / (float)imgInfo.modelHeight;
    LogDebug << "Number of objects found : " << detBoxes.size() << " "
             << "widthScale: " << widthScale << " "
             << "heightScale: " << heightScale;
    for (auto& detBoxe : detBoxes) {
        if ((detBoxe.prob <= scoreThresh_) || (detBoxe.classID < 0)) {
            continue;
        }
        ObjDetectInfo objInfo = {};
        objInfo.classId = (float)detBoxe.classID;
        objInfo.confidence = detBoxe.prob;

        objInfo.x0 = std::max<float>(detBoxe.x - detBoxe.width / COORDINATE_PARAM, 0) * widthScale;
        objInfo.y0 = std::max<float>(detBoxe.y - detBoxe.height / COORDINATE_PARAM, 0) * heightScale;
        objInfo.x1 = std::max<float>(detBoxe.x + detBoxe.width / COORDINATE_PARAM, 0) * widthScale;
        objInfo.y1 = std::max<float>(detBoxe.y + detBoxe.height / COORDINATE_PARAM, 0) * heightScale;
        LogDebug << "Find object: "
                 << "classId(" << objInfo.classId << "), confidence(" << objInfo.confidence << "), Coordinates("
                 << objInfo.x0 << ", " << objInfo.y0 << "; " << objInfo.x1 << ", " << objInfo.y1 << ").";
        objInfos.push_back(objInfo);
    }
}

/*
 * @description: Get the info of detected object from output and resize to
 * original coordinates.
 * @param featLayerData Vector of output feature data.
 * @param objInfos Address of output object infos.
 * @param imgInfo Info of model/image width and height.
 * @return: void
 */
void MSFasterRcnnPostProcessor::ObjectDetectionOutput(
    std::vector<std::shared_ptr<void>>& featLayerData,
    std::vector<ObjDetectInfo>& objInfos,
    ImageInfo& imgInfo) {
    LogDebug << "MSFasterRcnnPostProcessor start to write results.";

    std::vector<MxBase::DetectBox> detBoxes;
    GetValidDetBoxes(featLayerData, detBoxes, imgInfo);
    MxBase::NmsSort(detBoxes, iouThresh_, MxBase::MAX);
    ConvertObjInfoFromDetectBox(detBoxes, objInfos, imgInfo);

    LogDebug << "MSFasterRcnnPostProcessor write results successed.";
}

} // namespace MxBase

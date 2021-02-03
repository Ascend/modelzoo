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

#include "MSMaskRcnnPostProcess.h"

#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "acl/acl.h"

namespace {
// Output Tensor
const int OUTPUT_TENSOR_SIZE = 4;

const int OUTPUT_BBOX_INDEX = 0;
const int OUTPUT_CLASS_INDEX = 1;
const int OUTPUT_MASK_INDEX = 2;
const int OUTPUT_MASK_AREA_INDEX = 3;
}  // namespace

namespace {
const int LEFTTOPY = 0;
const int LEFTTOPX = 1;
const int RIGHTBOTY = 2;
const int RIGHTBOTX = 3;
}  // namespace

namespace MxBase {

APP_ERROR MSMaskRcnnPostProcessor::ReadConfigParams() {
    APP_ERROR ret = configData_.GetFileValue<int>("CLASS_NUM", classNum_);
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

    ret = configData_.GetFileValue<int>("RPN_MAX_NUM", rpnMaxNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No RPN_MAX_NUM in config file, default value(" << rpnMaxNum_ << ").";
    }

    ret = configData_.GetFileValue<int>("MAX_PER_IMG", maxPerImg_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MAX_PER_IMG in config file, default value(" << maxPerImg_ << ").";
    }

    ret = configData_.GetFileValue<float>("MASK_THREAD_BINARY", maskThrBinary_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MASK_THREAD_BINARY in config file, default value(" << maskThrBinary_ << ").";
    }

    ret = configData_.GetFileValue<int>("MASK_SHAPE_SIZE", maskSize_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MASK_SHAPE_SIZE in config file, default value(" << maskSize_ << ").";
    }

    ret = configData_.GetFileValue<bool>("SAVE_RESULT_TO_JSON", saveResultToJson_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No SAVE_RESULT_TO_JSON in config file, default value(" << saveResultToJson_
                << ").";
    }

    LogInfo << "The config parameters of post process are as follows: \n"
            << "  CLASS_NUM: " << classNum_ << " \n"
            << "  SCORE_THRESH: " << scoreThresh_ << " \n"
            << "  IOU_THRESH: " << iouThresh_ << " \n"
            << "  RPN_MAX_NUM: " << rpnMaxNum_ << " \n"
            << "  MAX_PER_IMG: " << maxPerImg_ << " \n"
            << "  MASK_THREAD_BINARY: " << maskThrBinary_ << " \n"
            << "  MASK_SHAPE_SIZE: " << maskSize_;
}

APP_ERROR
MSMaskRcnnPostProcessor::Init(const std::string &configPath, const std::string &labelPath,
                              MxBase::ModelDesc modelDesc) {
    LogInfo << "Begin to initialize MSMaskRcnnPostProcessor.";
    APP_ERROR ret = LoadConfigDataAndLabelMap(configPath, labelPath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to LoadConfigDataAndLabelMap in MSFasterRcnnPostProcessor.";
        return ret;
    }

    ReadConfigParams();

    GetModelTensorsShape(modelDesc);
    if (checkModelFlag_) {
        ret = CheckMSModelCompatibility();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to CheckModelCompatibility in MSMaskRcnnPostProcessor."
                     << "Please check the compatibility between model and postprocessor";
            return ret;
        }
    } else {
        LogWarn << "Compatibility check for model is skipped as CHECK_MODEL is set "
                   "as false, please ensure your model"
                << "is correct before running.";
    }

    LogInfo << "End to initialize MSMaskRcnnPostProcessor.";
    return APP_ERR_OK;
}

/*
 * @description: Do nothing temporarily.
 * @return: APP_ERROR error code.
 */
APP_ERROR MSMaskRcnnPostProcessor::DeInit() {
    LogInfo << "Begin to deinitialize MSMaskRcnnPostProcessor.";
    LogInfo << "End to deinitialize MSMaskRcnnPostProcessor.";
    return APP_ERR_OK;
}

APP_ERROR MSMaskRcnnPostProcessor::Process(std::vector<std::shared_ptr<void>> &featLayerData,
                                           std::vector<ObjDetectInfo> &objInfos, const bool useMpPictureCrop,
                                           MxBase::PostImageInfo postImageInfo) {
    LogDebug << "Begin to process MSMaskRcnnPostProcessor.";
    ObjectPostProcessorBase::Process(featLayerData, objInfos, useMpPictureCrop, postImageInfo);
    LogDebug << "End to process MSMaskRcnnPostProcessor.";
    return APP_ERR_OK;
}

void MSMaskRcnnPostProcessor::FreeMaskMemory(std::vector<ObjDetectInfo> &objInfos) {
    for (auto &obj : objInfos) {
        if (obj.maskPtr == nullptr) {
            continue;
        }
        free(obj.maskPtr);
        obj.maskPtr = nullptr;
    }
}

APP_ERROR MSMaskRcnnPostProcessor::CheckMSModelCompatibility() {
    if (outputTensorShapes_.size() != OUTPUT_TENSOR_SIZE) {
        LogError << "The size of output tensor is wrong, output size(" << outputTensorShapes_.size() << ")";
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_BBOX_INDEX].size() != 3) {
        LogError << "outputTensorShapes_[1].size(): " << outputTensorShapes_[OUTPUT_BBOX_INDEX].size();
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    int total_num = classNum_ * rpnMaxNum_;
    if (outputTensorShapes_[OUTPUT_BBOX_INDEX][1] != total_num) {
        LogError << "The output tensor is mismatched: (" << total_num << "/"
                 << outputTensorShapes_[OUTPUT_BBOX_INDEX][1] << "). "
                 << "Please check that the hyper-parameter(classNum_, rpnMaxNum_) "
                    "are configured correctly.";
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

    if (outputTensorShapes_[OUTPUT_MASK_AREA_INDEX][1] != total_num) {
        LogError << "outputTensorShapes_[OUTPUT_MASK_AREA_INDEX][1]: "
                 << outputTensorShapes_[OUTPUT_MASK_AREA_INDEX][1];
        return APP_ERR_OUTPUT_NOT_MATCH;
    }

    if (outputTensorShapes_[OUTPUT_MASK_AREA_INDEX][2] != maskSize_) {
        LogError << "The tensor of mask is mismatched: (" << outputTensorShapes_[OUTPUT_MASK_AREA_INDEX][3] << "/"
                 << maskSize_ << "). "
                 << "Please check that the hyper-parameter(MASK_SHAPE_SIZE) are "
                    "configured correctly.";
        return APP_ERR_OUTPUT_NOT_MATCH;
    }
    return APP_ERR_OK;
}

static bool CompareDetectBoxes(const MxBase::DetectBox &box1, const MxBase::DetectBox &box2) {
    return box1.prob > box2.prob;
}

static void GetDetectBoxesTopK(std::vector<MxBase::DetectBox> &detBoxes, size_t kVal) {
    std::sort(detBoxes.begin(), detBoxes.end(), CompareDetectBoxes);
    if (detBoxes.size() <= kVal) {
        return;
    }

    LogDebug << "Total detect boxes: " << detBoxes.size() << ", kVal: " << kVal;
    detBoxes.erase(detBoxes.begin() + kVal, detBoxes.end());
}

void MSMaskRcnnPostProcessor::GetValidDetBoxes(std::vector<std::shared_ptr<void>> &featLayerData,
                                               std::vector<MxBase::DetectBox> &detBoxes, ImageInfo &imgInfo) const {
    LogInfo << "Begin to GetValidDetBoxes Mask GetValidDetBoxes.";
    auto *bboxPtr = static_cast<aclFloat16 *>(featLayerData[OUTPUT_BBOX_INDEX].get());           // 1 * 80000 * 5
    auto *labelPtr = static_cast<int32_t *>(featLayerData[OUTPUT_CLASS_INDEX].get());            // 1 * 80000 * 1
    auto *maskPtr = static_cast<bool *>(featLayerData[OUTPUT_MASK_INDEX].get());                 // 1 * 80000 * 1
    auto *maskAreaPtr = static_cast<aclFloat16 *>(featLayerData[OUTPUT_MASK_AREA_INDEX].get());  // 1 * 80000 * 28 * 28
    // mask filter
    float prob;
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
        detBox.maskPtr = maskAreaPtr + index * maskSize_ * maskSize_;
        detBoxes.push_back(detBox);
    }
    GetDetectBoxesTopK(detBoxes, maxPerImg_);
}

APP_ERROR GetMaskSize(const ObjDetectInfo &objInfo, ImageInfo &imgInfo, cv::Size &maskSize) {
    int width = std::ceil(std::min<float>(objInfo.x1 - objInfo.x0 + 1, (float)imgInfo.imgWidth - objInfo.x0));
    int height = std::ceil(std::min<float>(objInfo.y1 - objInfo.y0 + 1, (float)imgInfo.imgHeight - objInfo.y0));
    if (width < 1 || height < 1) {
        LogError << "The mask bbox is invalid, will be ignored, bboxWidth: " << width << ", bboxHeight: " << height
                 << ".";
        return APP_ERR_COMM_FAILURE;
    }

    maskSize.width = width;
    maskSize.height = height;
    return APP_ERR_OK;
}

APP_ERROR MSMaskRcnnPostProcessor::MaskPostProcess(ObjDetectInfo &objInfo, void *maskPtr, ImageInfo &imgInfo) {
    // resize
    cv::Mat maskMat(maskSize_, maskSize_, CV_32FC1);
    // maskPtr aclFloat16 to float
    aclFloat16 *maskTempPtr;
    auto *maskAclPtr = reinterpret_cast<aclFloat16 *>(maskPtr);
    for (int row = 0; row < maskMat.rows; ++row) {
        maskTempPtr = maskAclPtr + row * maskMat.cols;
        for (int col = 0; col < maskMat.cols; ++col) {
            maskMat.at<float>(row, col) = aclFloat16ToFloat(*(maskTempPtr + col));
        }
    }

    cv::Size maskSize;
    APP_ERROR ret = MxBase::GetMaskSize(objInfo, imgInfo, maskSize);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    size_t bboxWidth = maskSize.width;
    size_t bboxHeight = maskSize.height;

    cv::Mat maskDst;
    cv::resize(maskMat, maskDst, cv::Size(bboxWidth, bboxHeight));
    bool *objMaskPtr = (bool *)malloc(sizeof(bool) * bboxWidth * bboxHeight);
    if (objMaskPtr == nullptr) {
        LogError << "Malloc memory failed for mask data.";
        return APP_ERR_COMM_ALLOC_MEM;
    }

    memset(objMaskPtr, false, sizeof(bool) * bboxWidth * bboxHeight);
    for (size_t row = 0; row < bboxHeight; ++row) {
        for (size_t col = 0; col < bboxWidth; ++col) {
            if (maskDst.at<float>(row, col) <= maskThrBinary_) {
                continue;
            }
            objMaskPtr[row * bboxWidth + col] = true;
        }
    }

    objInfo.maskPtr = (void *)objMaskPtr;
    return APP_ERR_OK;
}

void MSMaskRcnnPostProcessor::ConvertObjInfoFromDetectBox(std::vector<MxBase::DetectBox> &detBoxes,
                                                          std::vector<ObjDetectInfo> &objInfos, ImageInfo &imgInfo) {
    float widthScale = (float)imgInfo.imgWidth / (float)imgInfo.modelWidth;
    float heightScale = (float)imgInfo.imgHeight / (float)imgInfo.modelHeight;
    LogDebug << "Number of objects found : " << detBoxes.size() << " "
             << "widthScale: " << widthScale << " "
             << "heightScale: " << heightScale;

    APP_ERROR ret = APP_ERR_OK;
    for (auto &detBoxe : detBoxes) {
        if (detBoxe.classID < 0) {
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
        ret = MaskPostProcess(objInfo, detBoxe.maskPtr, imgInfo);
        if (ret == APP_ERR_COMM_FAILURE) {
            continue;
        } else if (ret != APP_ERR_OK) {
            break;
        }

        objInfos.push_back(objInfo);
    }

    if (ret != APP_ERR_OK) {
        LogError << "Convert obj info failed.";
        FreeMaskMemory(objInfos);
    }
}

APP_ERROR WriteResultToJson(const std::vector<ObjDetectInfo> &objInfos, ImageInfo &imgInfo) {
    if (objInfos.empty()) {
        LogWarn << "The predict result is empty.";
        return APP_ERR_OK;
    }
    namespace pt = boost::property_tree;
    pt::ptree root, data;
    APP_ERROR ret;
    for (auto &obj : objInfos) {
        pt::ptree item;
        item.put("classId", obj.classId);
        item.put("confidence", obj.confidence);
        item.put("x0", obj.x0);
        item.put("y0", obj.y0);
        item.put("x1", obj.x1);
        item.put("y1", obj.y1);
        cv::Size maskSize;
        ret = MxBase::GetMaskSize(obj, imgInfo, maskSize);
        if (ret != APP_ERR_OK) {
            return ret;
        }

        size_t width = maskSize.width;
        size_t height = maskSize.height;
        item.put("width", width);
        item.put("height", height);
        if (obj.maskPtr != nullptr) {
            std::string maskStr;
            auto maskPtr = reinterpret_cast<bool *>(obj.maskPtr);
            for (size_t j = 0; j < height; ++j) {
                size_t colPos = j * width;
                for (size_t z = 0; z < width; ++z) {
                    maskStr.append(maskPtr[colPos + z] ? "1" : "0");
                }
            }
            item.put("mask", maskStr);
        }
        data.push_back(std::make_pair("", item));
    }
    root.add_child("data", data);
    pt::json_parser::write_json(PREDICT_RESULT_PATH, root, std::locale(), false);
    return APP_ERR_OK;
}

/*
 * @description: Get the info of detected object from output and resize to
 * original coordinates.
 * @param featLayerData Vector of output feature data.
 * @param objInfos Address of output object infos.
 * @param imgInfo Info of model/image width and height.
 * @return: void
 */
void MSMaskRcnnPostProcessor::ObjectDetectionOutput(std::vector<std::shared_ptr<void>> &featLayerData,
                                                    std::vector<ObjDetectInfo> &objInfos, ImageInfo &imgInfo) {
    LogDebug << "MSMaskRcnnPostProcessor start to write results.";

    std::vector<MxBase::DetectBox> detBoxes;
    GetValidDetBoxes(featLayerData, detBoxes, imgInfo);
    MxBase::NmsSort(detBoxes, iouThresh_, MxBase::MAX);
    ConvertObjInfoFromDetectBox(detBoxes, objInfos, imgInfo);
    if (saveResultToJson_) {
        WriteResultToJson(objInfos, imgInfo);
        FreeMaskMemory(objInfos);
    }

    LogDebug << "MSMaskRcnnPostProcessor write results successed.";
}

}  // namespace MxBase

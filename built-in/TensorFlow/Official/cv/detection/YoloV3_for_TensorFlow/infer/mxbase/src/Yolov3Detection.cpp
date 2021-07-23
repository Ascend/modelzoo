/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Yolov3Detection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <unistd.h>
#include <sys/stat.h>


APP_ERROR Yolov3DetectionOpencv::LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap)
{
    std::ifstream infile;
    // open label file
    infile.open(labelPath, std::ios_base::in);
    std::string s;
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << labelPath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    labelMap.clear();
    // construct label map
    int count = 0;
    while (std::getline(infile, s)) {
        if (s.find('#') <= 1) {
            continue;
        }
        size_t eraseIndex = s.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos) {
            s.erase(eraseIndex + 1, s.size() - eraseIndex);
        }
        labelMap.insert(std::pair<int, std::string>(count, s));
        count++;
    }
    infile.close();
    return APP_ERR_OK;
}

void Yolov3DetectionOpencv::SetYolov3PostProcessConfig(const InitParam &initParam, std::map<std::string, std::shared_ptr<void>> &config)
{
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("BIASES_NUM", std::to_string(initParam.biasesNum));
    configData.SetJsonValue("BIASES", initParam.biases);
    configData.SetJsonValue("OBJECTNESS_THRESH", initParam.objectnessThresh);
    configData.SetJsonValue("IOU_THRESH", initParam.iouThresh);
    configData.SetJsonValue("SCORE_THRESH", initParam.scoreThresh);
    configData.SetJsonValue("YOLO_TYPE", std::to_string(initParam.yoloType));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
    configData.SetJsonValue("ANCHOR_DIM", std::to_string(initParam.anchorDim));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
}

APP_ERROR Yolov3DetectionOpencv::Init(const InitParam &initParam)
{
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
   
    std::map<std::string, std::shared_ptr<void>> config;
    SetYolov3PostProcessConfig(initParam, config);
    post_ = std::make_shared<MxBase::Yolov3PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    // load labels from file
    ret = LoadLabels(initParam.labelPath, labelMap_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

void Yolov3DetectionOpencv::ReadImage(const std::string &imgPath, cv::Mat &imageMat)
{
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    imageWidth_ = imageMat.cols;
    imageHeight_ = imageMat.rows;
    return;
}

void Yolov3DetectionOpencv::Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat)
{
    static constexpr uint32_t resizeHeight = 416;
    static constexpr uint32_t resizeWidth = 416;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return;
}

APP_ERROR Yolov3DetectionOpencv::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase)
{
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs)
{
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::PostProcess(const std::vector<MxBase::TensorBase> &outputs,
                                             std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = imageWidth_;
    imgInfo.heightOriginal = imageHeight_;
    imgInfo.widthResize = 416;
    imgInfo.heightResize = 416;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    APP_ERROR ret = post_->Process(outputs, objInfos, imageInfoVec);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::WriteResult(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
{
    std::string resultPathName = "result";
    uint32_t batchSize = objInfos.size();
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/result.txt";
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    for (uint32_t i = 0; i < batchSize; i++) {
        tfile << "Object detected num is " << objInfos[i].size() << std::endl;
        for (uint32_t j = 0; j < objInfos[i].size(); j++) {
            tfile << "#Obj: " << j << ", box: " << objInfos[i][j].x0 << " " << objInfos[i][j].y0 << " "
                << objInfos[i][j].x1 << " " << objInfos[i][j].y1
                << ", confidence: " << objInfos[i][j].confidence << ", lable: " << labelMap_[objInfos[i][j].classId]
                << ", id: " << objInfos[i][j].classId << std::endl;
        }
    }

    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR Yolov3DetectionOpencv::Process(const std::string &imgPath)
{
    // process image
    cv::Mat imageMat;
    ReadImage(imgPath, imageMat);

    Resize(imageMat, imageMat);
  
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
    ret = PostProcess(outputs, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    imageMat.release();
    return APP_ERR_OK;
}
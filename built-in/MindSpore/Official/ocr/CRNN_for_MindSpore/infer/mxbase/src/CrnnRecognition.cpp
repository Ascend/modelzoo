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
#include "CrnnRecognition.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;
namespace {
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR CrnnRecognition::Init(const InitParam &initParam)
{
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERROR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERROR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if ( ret != APP_ERROR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERROR_OK){
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string argmax = initParam.argmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("OBJECT_NUM", std::to_string(initParam.objectNum));
    configData.SetJsonValue("BLANK_INDEX", std::to_string(initParam.blankIndex));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("WITH_ARGMAX", argmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::share_ptr<void>> config;
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);

    post_ = std::make_shared<MxBase::CrnnPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERROR_OK){
        LogError << "CrnnPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERROR_OK;
}

APP_ERROR CrnnRecognition::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERROR_OK;
}

APP_ERROR CrnnRecognition::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor)
{
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERROR_OK) {
        LogError << "DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0){
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = TensorBase(memoryData, false, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor)
{
    auto shape = inputTensor.GetShape();
    for (auto i : shape){
        LogError << "inputTensor.GetShape() is " << i << ".";
    }
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    const uint32_t resizeHeight = 34;
    const uint32_t resizeWidth = 128;
    MxBase::ResizeConfig resize = {};
    resize.height = resizeHeight;
    resize.width = resizeWidth;
    MxBase::DvppDataInfo output = {};

    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::CropRoiConfig cropConfig = {
        .x0 = 0,
        .x1 = 100,
        .y1 = 32,
        .y0 = 0
    };
    MxBase::DvppDataInfo crop_output = {};

    APP_ERROR crop_net = dvppWrapper_->VpcCrop(output, crop_output, cropConfig);
    if (crop_ret != APP_ERR_OK) {
        LogError << "VpcCrop failed, ret=" << crop_net << ".";
        return crop_net;
    }

    MxBase::MemoryData memoryData((void*)crop_output.data, crop_output.dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (crop_output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << crop_output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    shape = {crop_output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, crop_output.widthStride};
    LogError << "output.height is " << output.height << ", output.width is " << output.width;
    LogError << "output.heightStride is " << output.heightStride << ", output.widthStride is " << output.widthStride;
    for (auto i : shape){
        LogError << "output shape is " << i << ".";
    }
    outputTensor = TensorBase(memoryData, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::ReadAndResize(const std::string &imgPath, MxBase::TensorBase &outputTensor)
{
    cv::Mat srcImageMat = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat dstImageMat;
    uint32_t resizeWidth = 100;
    uint32_t resizeHeight = 32;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));

    uint32_t dataSize = dstImageMat.cols * dstImageMat.rows;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(dstImageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {static_cast<uint32_t>(dstImageMat.rows), static_cast<uint32_t>(dstImageMat.cols)};
    outputTensor = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs)
{
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i){
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j){
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }

        int tmp_shape = 0;
        tmp_shape = shape[1];
        shape[1] = shape[0];
        shape[0] = tmp_shape;

        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }

        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicInfo::STATIC_BATCH;

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::PostProcess(const std::vector<MxBase::TensorBase>& tensors, std::vector<MxBase::TextsInfo>& textInfos)
{
    APP_ERROR ret = post_->Process(tensors, textInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::Process(const std::string &imgPath, std::string &result)
{
    TensorBase resizeImage;
    APP_ERROR ret = ReadAndResize(imgPath, resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Read and resize image failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(resizeImage);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TextsInfo> TextInfos = {};
    ret = PostProcess(outputs, TextInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t topkIndex = 1;
    for (auto textInfos : TextInfos) {
        if (topkIndex > 1) {
            break;
        }
        for (size_t i = 0; i < textInfos.text.size(); ++i) {
            LogDebug << " top" << topkIndex << " text: " << textInfos.text[i];
            result = textInfos.text[i];
        }
        topkIndex++;
    }
    return APP_ERR_OK;
}
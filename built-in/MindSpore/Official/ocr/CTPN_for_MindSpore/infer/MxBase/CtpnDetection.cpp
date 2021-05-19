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
#include "CtpnDetection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using namespace MxBase;
namespace {
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR CtpnDetection::Init(const InitParam &initParam)
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
    const std::string isOriented = initParam.IsOriented ? "true" : "false";
    const std::string isMindspore = initParam.IsMindspore ? "true" : "false";
    const std::string checkModelFlag = initParam.checkModelFlag ? "true" : "false";

    configData.SetJsonValue("MAX_HORIZONTAL_GAP", std::to_string(initParam.maxHorizontalGap));
    configData.SetJsonValue("BOX_IOU_THRESH", std::to_string(initParam.boxIouThresh));
    configData.SetJsonValue("TEXT_IOU_THRESH", std::to_string(initParam.TextIouThresh));
    configData.SetJsonValue("TEXT_PROPROSAL_MIN_SCORE", std::to_string(initParam.TextProposalsMinScore));
    configData.SetJsonValue("LINE_MIN_SCORE", std::to_string(initParam.LineMinScore));
    configData.SetJsonValue("IS_ORIENTED", IsOriented);
    configData.SetJsonValue("IS_MINDSPORE", IsMindspore);
    configData.SetJsonValue("CHECK_MODEL_FLAG", checkModelFlag);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::share_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::CtpnPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERROR_OK){
        LogError << "CtpnPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERROR_OK;
}

APP_ERROR CtpnDetection::DeInit()
{
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERROR_OK;
}

APP_ERROR CtpnDetection::ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor, int& oriWidth, int& oriHeight)
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

    oriWidth = output.width;
    oriHeight = output.height;
    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    tensor = TensorBase(memoryData, false, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CtpnDetection::Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor)
{
    auto shape = inputTensor.GetShape();
    MxBase::DvppDataInfo input = {};
    input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.width = shape[1];
    input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
    input.widthStride = shape[1];
    input.dataSize = inputTensor.GetByteSize();
    input.data = (uint8_t*)inputTensor.GetBuffer();
    const uint32_t resizeHeight = 576;
    const uint32_t resizeWidth = 960;
    MxBase::ResizeConfig resize = {};
    resize.height = resizeHeight;
    resize.width = resizeWidth;
    MxBase::DvppDataInfo output = {};
    APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
    if (ret != APP_ERR_OK) {
        LogError << "VpcResize failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::MemoryData memoryData((void*)output.data, output.dataSize, MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    if (output.heightStride % VPC_H_ALIGN != 0) {
        LogError << "Output data height(" << output.heightStride << ") can't be divided by " << VPC_H_ALIGN << ".";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};

    for (auto i : shape){
        LogError << "output shape is " << i << ".";
    }
    outputTensor = TensorBase(memoryData, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CtpnDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs)
{
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i){
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j){
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }

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

    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR CtpnDetection::PostProcess(const std::vector<MxBase::TensorBase>& tensors,
        std::vector<std::vector<MxBase::TextObjectInfo>>& textObjInfos,
        const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos)
{
    APP_ERROR ret = post_->Process(tensors, textObjInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CtpnDetection::Process(const std::string &imgPath)
{
    TensorBase image;
    int width;
    int height;
    APP_ERROR ret = ReadImage(imgPath, image, width, height);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    TensorBase resizeImage;
    ret = Resize(image, resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
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

    std::vector<std::vector<MxBase::TextObjectInfo>> textObjInfos = {};
    resizedImageInfo_.widthResize = 960;
    resizedImageInfo_.heightResize = 576;
    resizedImageInfo_.widthOriginal = width;
    resizedImageInfo_.heightOriginal = height;
    resizedImageInfo_.resizeType = MxBase::ResizeType::RESIZE_STRETCHING
    resizedImageInfo_.keepAspectRatioScaling = 0;
    for (size_t i = 0; i < 1000; ++i){
        resizedImageInfos.push_back(resizedImageInfo_);
    }

    ret = PostProcess(outputs, textObjInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t batchIndex = 0;
    for (auto objInfos : textObjInfos) {
        uint32_t topkIndex = 1;
        for (auto objInfo : objInfos) {
            LogDebug << "batchIndex:" << batchIndex << " top" << topkIndex << " result:" << objInfo.result
                     << " confidence:" << objInfo.confidence;
            topkIndex++;
        }
        batchIndex++;
    }
    return APP_ERR_OK;
}
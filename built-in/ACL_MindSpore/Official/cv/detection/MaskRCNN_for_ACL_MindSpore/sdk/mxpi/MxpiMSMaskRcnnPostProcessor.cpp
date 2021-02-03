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

#include "MxpiMSMaskRcnnPostProcessor.h"

/*
 * @description Load the configs and labels from the file.
 * @param labelPath config path and label path.
 * @return APP_ERROR error code.
 */
APP_ERROR MxpiMSMaskRcnnPostProcessor::Init(const std::string &configPath, const std::string &labelPath,
                                            MxBase::ModelDesc modelDesc) {
    LogInfo << "Begin to initialize MxpiMSMaskRcnnPostProcessor.";
    APP_ERROR ret = postProcessorInstance_.Init(configPath, labelPath, modelDesc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to init MxpiMSMaskRcnnPostProcessor.";
        return ret;
    }
    LogInfo << "End to initialize MxpiMSMaskRcnnPostProcessor.";
    return APP_ERR_OK;
}

/*
 * @description: Do nothing temporarily.
 * @return: APP_ERROR error code.
 */
APP_ERROR MxpiMSMaskRcnnPostProcessor::DeInit() {
    LogInfo << "Begin to deinitialize MxpiMSMaskRcnnPostProcessor.";
    LogInfo << "End to deinitialize MxpiMSMaskRcnnPostProcessor.";
    return APP_ERR_OK;
}

/*
 * @description: Postprocess of object detection.
 * @param: metaDataPtr Pointer of metadata.
 * @param: useMpPictureCrop Flag whether use crop before modelInfer.
 * @param: postImageInfoVec Width and height of model/image.
 * @param: headerVec header of image in same buffer.
 * @param: tensors Output tensors of modelInfer.
 * @return: APP_ERROR error code.
 */
APP_ERROR MxpiMSMaskRcnnPostProcessor::Process(std::shared_ptr<void> &metaDataPtr,
                                               MxBase::PostProcessorImageInfo postProcessorImageInfo,
                                               std::vector<MxTools::MxpiMetaHeader> &headerVec,
                                               std::vector<std::vector<MxBase::BaseTensor>> &tensors) {
    APP_ERROR ret = MxPlugins::MxpiObjectPostProcessorBase::Process(metaDataPtr, postProcessorImageInfo, headerVec,
                                                                    tensors, postProcessorInstance_);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to Process MxpiMSMaskRcnnPostProcessor.";
        return ret;
    }
    return APP_ERR_OK;
}

std::shared_ptr<MxPlugins::MxpiModelPostProcessorBase> GetInstance() {
    LogInfo << "Begin to get MxpiMSMaskRcnnPostProcessor instance.";
    auto instance = std::make_shared<MxpiMSMaskRcnnPostProcessor>();
    LogInfo << "End to get MxpiMSMaskRcnnPostProcessor instance.";
    return instance;
}

/* *
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "inference_engine.h"
#include "asyn_inference_share_weight_engine.h"


Asyn_Inference_ShareWeight_engine::Asyn_Inference_ShareWeight_engine() : Asyn_InferEngine() {}

Asyn_Inference_ShareWeight_engine::Asyn_Inference_ShareWeight_engine(Config *config) : Asyn_InferEngine(config) {}

Asyn_Inference_ShareWeight_engine::Asyn_Inference_ShareWeight_engine(
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue, BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
    : Asyn_InferEngine(inQue, outQue)
{}

aclError Asyn_Inference_ShareWeight_engine::LoadModel()
{
    uint32_t modelSize = 0;
    aclError ret;

    ret = aclrtSetCurrentContext(this->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Set context failed");
        return 1;
    }

    modelData_ = nullptr;
    modelData_ = SdkInferReadBinFile(cfg_->om, modelSize);
    if (modelData_ == nullptr) {
        LOG_ERROR("SdkInferReadBinFile failed, ret %d", ret);
        return ret;
    }

    dev_ptr_ = nullptr;
    ret = aclrtMalloc(&dev_ptr_, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_ERROR_NONE != ret) {
        UnloadModel();
        LOG_ERROR("alloc dev_ptr_ failed, ret %d", ret);
        return ret;
    }

    modelId_ = 0;
    ret = aclmdlLoadFromMemWithMem(modelData_, modelSize, &modelId_, dev_ptr_, memSize, weight_ptr_, weightsize);
    if (ACL_ERROR_NONE != ret) {
        UnloadModel();
        LOG_ERROR("load model from memory failed, ret %d", ret);
        return ret;
    }

    LOG_INFO("Load model success. memSize: %lu, weightSize: %lu.", memSize, weightsize);

    modelDesc_ = nullptr;
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        LOG_ERROR("create model desc failed");
        UnloadModel();
        return 1;
    }

    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("get model desc failed, ret", ret);
        UnloadModel();
        return ret;
    }

    return ACL_ERROR_NONE;
}


aclError Asyn_Inference_ShareWeight_engine::Init(Config *config)
{
    aclError ret;
    dev_ptr_ = nullptr;
    weight_ptr_ = nullptr;
    modelData_ = nullptr;
    modelDesc_ = nullptr;
    modelId_ = 0;

    cfg_ = config;

    isTransToNextThread_ = false;

    ret = aclrtSetCurrentContext(cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return ret;
    }

    ret = LoadModel();
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("[ERROR]LoadModel failed, ret %d", ret);
        return ret;
    }

    ret = GetModelInputOutputInfo();
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("[ERROR]GetModelInputOutputInfo failed, ret %d", ret);
        UnloadModel();
        return ret;
    }

    int fileNum = SdkInferScanFiles(files_, cfg_->inputFolder);
    LOG_INFO("[INFO] file num : %d", fileNum);

    ret = InitImgStdValue(cfg_->resnetStdFile);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("[ERROR]InitImgStdValue failed, ret %d", ret);
        UnloadModel();
        return ret;
    }

    ret = InitYolov3ImgInfo(cfg_->yoloImgInfoFile);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("[ERROR]InitImgStdValue failed, ret %d", ret);
        UnloadModel();
        return ret;
    }

    ret = InitInferenceMemPool();
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]InitInferenceMemPool failed, ret[%d]", ret);
        UnloadModel();
        return ret;
    }

    return 0;
}

Asyn_Inference_ShareWeight_engine::~Asyn_Inference_ShareWeight_engine()
{
    LOG_INFO("e2e_Inference_engine deconstruct.");
}

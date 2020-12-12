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
#include "asyn_inference_with_syn_engine.h"

extern Asyn_Inference_with_syn_engine *Infer;
extern std::vector<aclrtContext> contex_vec;

extern void AsynInferWithDeviceSwitchThread(Asyn_Inference_with_syn_engine *inferEngine);

static void ModelExe_callback(void *userData)
{
    aclError ret;
    if (NULL == userData) {
        LOG_ERROR("[ERROR]callback func input para is null");
        return;
    }

    ModelExe_callback_data *pCallBackData = (ModelExe_callback_data *)userData;
    uint32_t start_index = pCallBackData->start_index;
    uint32_t end_index = pCallBackData->end_index;
    Asyn_Inference_with_syn_engine *inferEng = (Asyn_Inference_with_syn_engine *)pCallBackData->inferEng;

    aclrtSetCurrentContext(inferEng->cfg_->context);

    SdkInferGetTimeStart(&inferEng->timeCost_, POST_PROCESS);

    uint32_t memPoolSize = inferEng->cfg_->asynInferCfg.mem_pool_size;
    for (uint32_t i = start_index; i < end_index; i++) {
        aclmdlDataset *output = inferEng->inferMemPool_vec_[i % memPoolSize].output;
        std::vector<std::string> *inferFile_vec = inferEng->inferMemPool_vec_[i % memPoolSize].inferFile_vec;

        int retVal = 0;

        retVal = inferEng->SaveInferResult(output, inferFile_vec);
        if (retVal != 0) {
            printf("SaveInferResult fail, ret %d\n", ret);
        }
    }

    delete pCallBackData;

    SdkInferGetTimeEnd(&inferEng->timeCost_, POST_PROCESS);

    printf("[INFO] ModelExe_callback finished return\n");
}

aclError StartAysnInference(Asyn_Inference_with_syn_engine *inferEngine)
{
    aclError ret;
    int cnt = 0;
    int batchSize = inferEngine->cfg_->batchSize;
    std::vector<std::string> *inferFile_vec;
    uint32_t memPoolSize = inferEngine->cfg_->asynInferCfg.mem_pool_size;
    inferEngine->synStreamFinish_ = 0;

    for (int i = 0; i < inferEngine->files_.size(); i++) {
        if (cnt % batchSize == 0) {
            inferFile_vec = inferEngine->inferMemPool_vec_[inferEngine->memCurrent_ % memPoolSize].inferFile_vec;
            inferFile_vec->clear();
        }

        cnt++;
        inferFile_vec->push_back(inferEngine->files_[i]);

        if (cnt % batchSize == 0) {
            inferEngine->CreateInferInput(*inferFile_vec);

            ret = inferEngine->ExecInference();
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]AsynInferenceExecute failed");
            }

            if (inferEngine->memCurrent_ % memPoolSize == 0) {
                SdkInferGetTimeStart(&inferEngine->timeCost_, SYN_STREAM);
                ret = aclrtSynchronizeStream(inferEngine->inferStream_);
                SdkInferGetTimeEnd(&inferEngine->timeCost_, SYN_STREAM);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]aclrtSynchronizeStream failed, modelid=%d, line=%d", inferEngine->modelId_,
                        __LINE__);
                }

                LOG_INFO("mem pool is full, need call aclrtSynchronizeStream currentIndex %d",
                    inferEngine->memCurrent_);
            }
        }
    }

    if (cnt % batchSize != 0) {
        inferEngine->CreateInferInput(*inferFile_vec);

        ret = inferEngine->ExecInference();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]AsynInferenceExecute failed");
        }
    }

    if (inferEngine->memPre_ < inferEngine->memCurrent_) {
        ModelExe_callback_data *callbackData = new ModelExe_callback_data();
        callbackData->start_index = inferEngine->memPre_;
        callbackData->end_index = inferEngine->memCurrent_;
        callbackData->inferEng = (void *)inferEngine;

        SdkInferGetTimeStart(&inferEngine->timeCost_, LAUNCH_CALL_BACK);
        ret =
            aclrtLaunchCallback(ModelExe_callback, (void *)callbackData, ACL_CALLBACK_BLOCK, inferEngine->inferStream_);
        SdkInferGetTimeEnd(&inferEngine->timeCost_, LAUNCH_CALL_BACK);

        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]aclrtLaunchCallback fail, ret %d", ret);
        }

        inferEngine->memPre_ = inferEngine->memCurrent_;

        LOG_INFO("[INFO] last call aclrtLaunchCallback sucess");
    }

    SdkInferGetTimeStart(&inferEngine->timeCost_, SYN_STREAM);
    ret = aclrtSynchronizeStream(inferEngine->inferStream_);
    SdkInferGetTimeEnd(&inferEngine->timeCost_, SYN_STREAM);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtSynchronizeStream failed, modelid=%d, line=%d", inferEngine->modelId_, __LINE__);
    }

    inferEngine->synStreamFinish_ = 1;
}

Asyn_Inference_with_syn_engine::Asyn_Inference_with_syn_engine() : Asyn_InferEngine()
{
    memCurrent_ = 0;
    memPre_ = 0;
}

Asyn_Inference_with_syn_engine::Asyn_Inference_with_syn_engine(Config *config) : Asyn_InferEngine(config)
{
    memCurrent_ = 0;
    memPre_ = 0;
}

Asyn_Inference_with_syn_engine::Asyn_Inference_with_syn_engine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
    : Asyn_InferEngine(inQue, outQue)
{
    memCurrent_ = 0;
    memPre_ = 0;
}

static void *ThreadFunc(void *arg)
{
    Asyn_Inference_with_syn_engine *inferEng = (Asyn_Inference_with_syn_engine *)arg;
    aclrtSetCurrentContext(inferEng->cfg_->context);

    LOG_INFO("[INFO]thread start ");
    while (1) {
        // Notice: timeout 5000ms
        aclError aclRet = aclrtProcessReport(5000);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("[INFO]aclrtProcessReport timeout, ret = %d", aclRet);
        }

        if (inferEng->runFlag_ == 0) {
            LOG_INFO("[INFO]ThreadFunc exit..............");
            break;
        }
    }

    return NULL;
}

aclError Asyn_Inference_with_syn_engine::InferenceThreadProc()
{
    std::thread inferThread(AsynInferWithDeviceSwitchThread, this);
    threads_.push_back(std::move(inferThread));

    return ACL_ERROR_NONE;
}

aclError Asyn_Inference_with_syn_engine::ExecInference()
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t memPoolSize = cfg_->asynInferCfg.mem_pool_size;

    aclmdlDataset *input = inferMemPool_vec_[memCurrent_ % memPoolSize].input;
    std::vector<std::string> *inferFile_vec = inferMemPool_vec_[memCurrent_ % memPoolSize].inferFile_vec;

    if (cfg_->modelType.compare("yolov3") == 0) {
        ret = CreateYoloImageInfoInput(input, inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("YoloInferInputInit fail, ret[%d]", ret);
            return ret;
        }
    }

    aclmdlDataset *output = inferMemPool_vec_[memCurrent_ % memPoolSize].output;

    SdkInferGetTimeStart(&timeCost_, ASYN_MODEL_EXECUTE);
    ret = aclmdlExecuteAsync(modelId_, input, output, inferStream_);
    SdkInferGetTimeEnd(&timeCost_, ASYN_MODEL_EXECUTE);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclmdlExecuteAsync failed, ret");
        return ret;
    }

    memCurrent_++;

    if ((memCurrent_ % cfg_->asynInferCfg.callback_interval == 0) && (memCurrent_ > 0)) {
        ModelExe_callback_data *callbackData = new ModelExe_callback_data();
        callbackData->start_index = memPre_;
        callbackData->end_index = memCurrent_;
        callbackData->inferEng = (void *)this;

        SdkInferGetTimeStart(&timeCost_, LAUNCH_CALL_BACK);
        ret = aclrtLaunchCallback(ModelExe_callback, (void *)callbackData, ACL_CALLBACK_BLOCK, inferStream_);
        SdkInferGetTimeEnd(&timeCost_, LAUNCH_CALL_BACK);

        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]aclrtLaunchCallback fail, ret %d", ret);
        }

        memPre_ = memCurrent_;
    }

    return ret;
}

void Asyn_Inference_with_syn_engine::UnSubscribeAndDtyStream()
{
    aclError ret;
    ret = aclrtSetCurrentContext(cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed, ret %d", ret);
        return;
    }
    // unsubscribe report
    ret = aclrtUnSubscribeReport(static_cast<uint64_t>(callbackPid_), inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]acl unsubscribe report failed");
    }

    runFlag_ = 0;
    (void)pthread_join(static_cast<uint64_t>(callbackPid_), nullptr);

    ret = aclrtDestroyStream(inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtDestroyStream failed");
    }
}

/*
 * @brief thread entry of asyn inference with device switch
 * @param [in] inferEngine: the pointer of asyn inference obj
 * @return none
 */
void AsynInferWithDeviceSwitchThread(Asyn_Inference_with_syn_engine *inferEngine)
{
    aclError ret;

    ret = aclrtSetCurrentContext(inferEngine->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed, ret %d", ret);
        return;
    }
    LOG_INFO("[INFO]aclrtSetCurrentContext success");

    int32_t curDeviceId;
    ret = aclrtGetDevice(&curDeviceId);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Get current deviceId failed, ret %d", ret);
        return;
    }

    ret = aclrtCreateStream(&inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]create stream failed");
    }
    inferEngine->runFlag_ = 1;
    ret = pthread_create(&inferEngine->callbackPid_, nullptr, ThreadFunc, (void *)inferEngine);
    if (ret != 0) {
        LOG_ERROR("[ERROR]create thread failed, err = %d", ret);
    }
    LOG_INFO("[INFO]pthread_create  success ");

    ret = aclrtSubscribeReport(static_cast<uint64_t>(inferEngine->callbackPid_), inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtSubscribeReport fail, ret %d", ret);
    }
    LOG_INFO("[INFO]aclrtSubscribeReport success");

    int cnt = 0;
    int batchSize = inferEngine->cfg_->batchSize;

    uint32_t loopCnt = 0;
    std::vector<std::string> *inferFile_vec;
    uint32_t memPoolSize = inferEngine->cfg_->asynInferCfg.mem_pool_size;

    while (loopCnt < inferEngine->cfg_->loopNum) {
        printf("loop index:%d\n", loopCnt);
        StartAysnInference(inferEngine);

        /*
        * swithch to another device,exec infer base this device infer engine
        */
        if ((curDeviceId == inference_json_cfg_tbl.commCfg.device_id_vec[0]) &&
            (inferEngine->cfg_->modelType.compare(0, 4, "yolo")) == 0) {
            LOG_INFO("current dev %d current channel %d current deviceId %d modelType %s switch device!!!",
                inferEngine->curDevIndex_, inferEngine->curChnIndex_, curDeviceId,
                inferEngine->cfg_->modelType.c_str());
            /*
            * switch to another device
            */
            ret = aclrtSetCurrentContext(contex_vec[1]);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Set infer context failed, ret %d", ret);
                return;
            }
            /*
            * use another device infergine exec infer
            */
            Asyn_Inference_with_syn_engine *otherInferEng =
                &Infer[1 * inference_json_cfg_tbl.inferCfg.channelNum * 2 + inferEngine->curChnIndex_];

            while (!otherInferEng->synStreamFinish_) {
                printf("other engine not finish syn stream, wait some ms\n");
                usleep(100000);
            }

            StartAysnInference(otherInferEng);

            LOG_INFO(
                "In new device finish infer new device index %d new channel index %d  %d modelType %s switch device!!!",
                otherInferEng->curDevIndex_, otherInferEng->curChnIndex_, otherInferEng->cfg_->modelType.c_str());
            /*
            * swith back to origin device
            */
            ret = aclrtSetCurrentContext(contex_vec[0]);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Set infer context failed, ret %d", ret);
                return;
            }
        }

        loopCnt++;
    }
}


Asyn_Inference_with_syn_engine::~Asyn_Inference_with_syn_engine()
{
    LOG_INFO("Asyn_Inference_with_syn_engine deconstruct.");
}

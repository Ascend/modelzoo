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
#include "asyn_infernce_with_waitevent_engine.h"


aclrtEvent event_;
bool c_flag = false;
std::vector<aclrtEvent> event_vec;
std::vector<bool> c_flag_vec;
uint32_t channelNum = inference_json_cfg_tbl.inferCfg.channelNum;

void AsynWaitEventInferThread(Asyn_Waitevent_InferEngine *inferEngine);

Asyn_Waitevent_InferEngine::Asyn_Waitevent_InferEngine() : Asyn_InferEngine()
{
    memCurrent_ = 0;
    memPre_ = 0;

    aclError ret = aclrtMemset(&timeCost_, sizeof(timeCost_), 0, sizeof(timeCost_));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtMemset timeCost_ failed");
    }
}

Asyn_Waitevent_InferEngine::Asyn_Waitevent_InferEngine(Config *config) : Asyn_InferEngine(config)
{
    memCurrent_ = 0;
    memPre_ = 0;

    aclError ret = aclrtMemset(&timeCost_, sizeof(timeCost_), 0, sizeof(timeCost_));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtMemset timeCost_ failed");
    }
}

static void ModelExe_callback(void *userData)
{
    printf("+++++++++++++++++++++++start ModelExe_callback++++++++++++++++++++++ \n");
    sleep(3);
    aclError ret;
    if (NULL == userData) {
        LOG_ERROR("[ERROR]callback func input para is null\n");
        return;
    }

    ModelExe_callback_data *pCallBackData = (ModelExe_callback_data *)userData;
    uint32_t start_index = pCallBackData->start_index;
    uint32_t end_index = pCallBackData->end_index;
    Asyn_Waitevent_InferEngine *inferEng = (Asyn_Waitevent_InferEngine *)pCallBackData->inferEng;

    aclrtSetCurrentContext(inferEng->cfg_->context);

    SdkInferGetTimeStart(&inferEng->timeCost_, POST_PROCESS);

    uint32_t memPoolSize = inferEng->cfg_->asynInferCfg.mem_pool_size;
    for (uint32_t i = start_index; i < end_index; i++) {
        aclmdlDataset *output = inferEng->inferMemPool_vec_[i % memPoolSize].output;
        std::vector<std::string> *inferFile_vec = inferEng->inferMemPool_vec_[i % memPoolSize].inferFile_vec;
        LOG_INFO(
            "[INFO]ModelExe_callback start_index %d end_index %d  memPool index %d, output = %p, inferFile_vec = %p\n",
            start_index, end_index, (i % memPoolSize), output, inferFile_vec);
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

static void *ThreadFunc(void *arg)
{
    Asyn_Waitevent_InferEngine *inferEng = (Asyn_Waitevent_InferEngine *)arg;
    aclrtSetCurrentContext(inferEng->cfg_->context);

    LOG_INFO("[INFO]thread start \n");
    while (1) {
        /** Notice: timeout 1000ms
        */
        aclError aclRet = aclrtProcessReport(5000);
        if (aclRet != ACL_ERROR_NONE) {
            LOG_ERROR("[INFO]aclrtProcessReport timeout, ret = %d\n", aclRet);
        }

        if (inferEng->runFlag_ == 0) {
            LOG_INFO("[INFO]ThreadFunc exit..............\n");
            break;
        }
    }

    return NULL;
}

aclError Asyn_Waitevent_InferEngine::InferenceThreadProc_w()
{
    std::thread inferThread(AsynWaitEventInferThread, this);
    threads_.push_back(std::move(inferThread));
    return ACL_ERROR_NONE;
}

aclError Asyn_Waitevent_InferEngine::ExecInference_w(aclrtStream inferStream)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t memPoolSize = cfg_->asynInferCfg.mem_pool_size;

    aclmdlDataset *input = inferMemPool_vec_[memCurrent_ % memPoolSize].input;
    std::vector<std::string> *inferFile_vec = inferMemPool_vec_[memCurrent_ % memPoolSize].inferFile_vec;

    if (cfg_->modelType.compare("yolov3") == 0) {
        ret = CreateYoloImageInfoInput(input, inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("YoloInferInputInit fail, ret[%d]\n", ret);
            return ret;
        }
    }

    aclmdlDataset *output = inferMemPool_vec_[memCurrent_ % memPoolSize].output;

    SdkInferGetTimeStart(&timeCost_, ASYN_MODEL_EXECUTE);
    ret = aclmdlExecuteAsync(modelId_, input, output, inferStream);
    SdkInferGetTimeEnd(&timeCost_, ASYN_MODEL_EXECUTE);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclmdlExecuteAsync failed, ret\n");
        return ret;
    }

    memCurrent_++;

    if ((memCurrent_ % cfg_->asynInferCfg.callback_interval == 0) && (memCurrent_ > 0)) {
        ModelExe_callback_data *callbackData = new ModelExe_callback_data();
        callbackData->start_index = memPre_;
        callbackData->end_index = memCurrent_;
        callbackData->inferEng = this;

        SdkInferGetTimeStart(&timeCost_, LAUNCH_CALL_BACK);
        ret = aclrtLaunchCallback(ModelExe_callback, (void *)callbackData, ACL_CALLBACK_BLOCK, inferStream);
        SdkInferGetTimeEnd(&timeCost_, LAUNCH_CALL_BACK);

        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]aclrtLaunchCallback fail, ret %d\n", ret);
        }

        memPre_ = memCurrent_;

        LOG_INFO("[INFO]Index %d call aclrtLaunchCallback sucess\n",
            memCurrent_ / cfg_->asynInferCfg.callback_interval);
    }

    return ret;
}


void AsynWaitEventInferThread(Asyn_Waitevent_InferEngine *inferEngine)
{
    aclError ret = aclrtSetCurrentContext(inferEngine->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed\n");
        return;
    }
    LOG_INFO("[INFO]aclrtSetCurrentContext success \n");

    ret = aclrtCreateStream(&inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]create stream failed");
    }

    ret = aclrtCreateEvent(&event_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR] create event failed \n");
    }
    if (inferEngine->threadindex + 1 > event_vec.size()) {
        event_vec.push_back(event_);
    } else {
        event_vec.insert(event_vec.begin() + inferEngine->threadindex, event_);
    }

    while (true) {
        if (event_vec.size() == inferEngine->threadnum) {
            break;
        }
    }
    inferEngine->runFlag_ = 1;
    ret = pthread_create(&inferEngine->pid, nullptr, ThreadFunc, (void *)inferEngine);
    if (ret != 0) {
        LOG_ERROR("[ERROR]create thread failed, err = %d\n", ret);
    }
    LOG_INFO("[INFO]pthread_create  success \n");

    /** Specifies the thread that processes the callback function on the stream
     */
    ret = aclrtSubscribeReport(static_cast<uint64_t>(inferEngine->pid), inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtSubscribeReport fail, ret %d\n", ret);
    }
    LOG_INFO("[INFO]aclrtSubscribeReport success\n");
    if (c_flag_vec.size() == 0) {
        for (int n = 0; n < inferEngine->threadnum; n++) {
            c_flag_vec.push_back(false);
        }
    }
    sleep(1);
    int cnt = 0;
    int batchSize = inferEngine->cfg_->batchSize;
    uint32_t loopCnt = 0;
    void *p_batchDst = NULL;
    std::vector<std::string> *inferFile_vec;
    uint32_t memPoolSize = inferEngine->cfg_->asynInferCfg.mem_pool_size;
    sleep(1);

    while (loopCnt < inferEngine->cfg_->loopNum) {
        for (int i = 0; i < inferEngine->files_.size(); i++) {
            if (cnt % batchSize == 0) {
                inferFile_vec = inferEngine->inferMemPool_vec_[inferEngine->memCurrent_ % memPoolSize].inferFile_vec;
                inferFile_vec->clear();
            }
            cnt++;
            inferFile_vec->push_back(inferEngine->files_[i]);

            if (cnt % batchSize == 0) {
                inferEngine->CreateInferInput(*inferFile_vec);

                LOG_INFO("[INFO]Index %d inference execute!\n", inferEngine->memCurrent_);

                ret = inferEngine->ExecInference();
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]AsynInferenceExecute failed\n");
                }
                if (inferEngine->threadindex == 0) {
                    while (true) {
                        if (c_flag_vec[inferEngine->threadindex] == false) {
                            aclrtRecordEvent(event_vec[inferEngine->threadindex], inferEngine->inferStream_);
                            c_flag_vec[inferEngine->threadindex] = true;
                            break;
                        }
                    }
                } else if (inferEngine->threadindex != 0) {
                    while (true) {
                        if (c_flag_vec[inferEngine->threadindex] == false &&
                            c_flag_vec[inferEngine->threadindex - 1] == true) {
                            aclrtRecordEvent(event_vec[inferEngine->threadindex], inferEngine->inferStream_);
                            c_flag_vec[inferEngine->threadindex] = true;
                            break;
                        }
                    }
                    while (true) {
                        if (c_flag_vec[inferEngine->threadindex - 1] == true &&
                            c_flag_vec[inferEngine->threadindex] == true) {
                            aclrtStreamWaitEvent(inferEngine->inferStream_, event_vec[inferEngine->threadindex - 1]);
                            c_flag_vec[inferEngine->threadindex - 1] = false;
                            if (inferEngine->threadindex == inferEngine->threadnum - 1) {
                                c_flag_vec[inferEngine->threadindex] = false;
                            }
                            break;
                        }
                    }
                }
                if (inferEngine->memCurrent_ % memPoolSize == 0) {
                    SdkInferGetTimeStart(&inferEngine->timeCost_, SYN_STREAM);
                    ret = aclrtSynchronizeEvent(event_vec[inferEngine->threadindex]);

                    SdkInferGetTimeEnd(&inferEngine->timeCost_, SYN_STREAM);
                    if (ret != ACL_ERROR_NONE) {
                        LOG_ERROR("[ERROR]aclrtSynchronizeStream failed, modelid=%d, line=%d\n", inferEngine->modelId_,
                            __LINE__);
                    }
                    LOG_INFO("mem pool is full, need call aclrtSynchronizeStream currentIndex %d\n",
                        inferEngine->memCurrent_);
                }
            }
        }

        loopCnt++;
    }
    if (cnt % batchSize != 0) {
        inferEngine->CreateInferInput(*inferFile_vec);

        ret = inferEngine->ExecInference();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]AsynInferenceExecute failed\n");
        }
        if (inferEngine->threadindex == 0) {
            while (true) {
                if (c_flag_vec[inferEngine->threadindex] == false) {
                    aclrtRecordEvent(event_vec[inferEngine->threadindex], inferEngine->inferStream_);
                    c_flag_vec[inferEngine->threadindex] = true;
                    break;
                }
            }
        } else if (inferEngine->threadindex != 0) {
            while (true) {
                if (c_flag_vec[inferEngine->threadindex] == false && c_flag_vec[inferEngine->threadindex - 1] == true) {
                    aclrtRecordEvent(event_vec[inferEngine->threadindex], inferEngine->inferStream_);
                    c_flag_vec[inferEngine->threadindex] = true;
                    break;
                }
            }
            while (true) {
                if (c_flag_vec[inferEngine->threadindex - 1] == true && c_flag_vec[inferEngine->threadindex] == true) {
                    aclrtStreamWaitEvent(inferEngine->inferStream_, event_vec[inferEngine->threadindex - 1]);
                    c_flag_vec[inferEngine->threadindex - 1] = false;
                    if (inferEngine->threadindex == inferEngine->threadnum - 1) {
                        c_flag_vec[inferEngine->threadindex] = false;
                    }
                    break;
                }
            }
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
            LOG_ERROR("[ERROR]aclrtLaunchCallback fail, ret %d\n", ret);
        }

        inferEngine->memPre_ = inferEngine->memCurrent_;

        LOG_INFO("[INFO] last call aclrtLaunchCallback sucess\n");
    }

    SdkInferGetTimeStart(&inferEngine->timeCost_, SYN_STREAM);
    aclrtSynchronizeEvent(event_);

    SdkInferGetTimeEnd(&inferEngine->timeCost_, SYN_STREAM);

    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtSynchronizeStream failed, modelid=%d, line=%d\n", inferEngine->modelId_, __LINE__);
    }

    ret = aclrtUnSubscribeReport(static_cast<uint64_t>(inferEngine->pid), inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]acl unsubscribe report failed");
    }

    inferEngine->runFlag_ = 0;
    (void)pthread_join(static_cast<uint64_t>(inferEngine->pid), nullptr);

    if (inferEngine->threadindex != 0) {
        ret = aclrtDestroyEvent(event_vec[inferEngine->threadindex - 1]);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]aclrtDestroyEvent failed\n");
        }
        if (inferEngine->threadindex == inferEngine->threadnum - 1) {
            aclrtDestroyEvent(event_vec[inferEngine->threadindex]);
        }
    }

    ret = aclrtDestroyStream(inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtDestroyStream failed\n");
    }
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%d \n", inferEngine->threadindex);
}


Asyn_Waitevent_InferEngine::~Asyn_Waitevent_InferEngine()
{
    LOG_INFO("Asyn_Waitevent_InferEngine deconstruct.\n");
}
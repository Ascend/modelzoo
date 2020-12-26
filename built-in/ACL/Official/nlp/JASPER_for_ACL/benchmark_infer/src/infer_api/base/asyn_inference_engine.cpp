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

void AsynInferThread(Asyn_InferEngine *inferEngine);

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
    Asyn_InferEngine *inferEng = (Asyn_InferEngine *)pCallBackData->inferEng;

    aclrtSetCurrentContext(inferEng->cfg_->context);

    SdkInferGetTimeStart(&inferEng->timeCost_, POST_PROCESS);

    uint32_t memPoolSize = inferEng->cfg_->asynInferCfg.mem_pool_size;
    for (uint32_t i = start_index; i < end_index; i++) {
        aclmdlDataset *output = inferEng->inferMemPool_vec_[i % memPoolSize].output;
        std::vector<std::string> *inferFile_vec = inferEng->inferMemPool_vec_[i % memPoolSize].inferFile_vec;
        LOG_INFO(
            "[INFO]ModelExe_callback start_index %d end_index %d  memPool index %d, output = %p, inferFile_vec = %p",
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

Asyn_InferEngine::Asyn_InferEngine() : InferEngine()
{
    memCurrent_ = 0;
    memPre_ = 0;
}

Asyn_InferEngine::Asyn_InferEngine(Config *config) : InferEngine(config)
{
    memCurrent_ = 0;
    memPre_ = 0;
}

Asyn_InferEngine::Asyn_InferEngine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
    : InferEngine(inQue, outQue)
{
    memCurrent_ = 0;
    memPre_ = 0;
}

static void *ThreadFunc(void *arg)
{
    Asyn_InferEngine *inferEng = (Asyn_InferEngine *)arg;
    aclrtSetCurrentContext(inferEng->cfg_->context);

    LOG_INFO("[INFO]thread start");
    while (1) {
        // Notice: timeout 5ms
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

aclError Asyn_InferEngine::InferenceThreadProc()
{
    std::thread inferThread(AsynInferThread, this);
    threads_.push_back(std::move(inferThread));

    return ACL_ERROR_NONE;
}

aclError Asyn_InferEngine::Init(Config *config)
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

aclError Asyn_InferEngine::InitInferenceMemPool()
{
    aclError ret;

    ret = aclrtSetCurrentContext(this->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return ret;
    }

    size_t inputNum = aclmdlGetNumInputs(modelDesc_);
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);

    for (uint32_t i = 0; i < cfg_->asynInferCfg.mem_pool_size; i++) {
        aclmdlDataset *input = aclmdlCreateDataset();

        for (size_t inputIndex = 0; inputIndex < inputNum; inputIndex++) {
            size_t size = aclmdlGetInputSizeByIndex(modelDesc_, inputIndex);
            void *dst;
            ret = aclrtMalloc(&dst, size, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Malloc device failed, ret[%d]", ret);
                return ret;
            }

            aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, size);
            if (inputData == nullptr) {
                LOG_ERROR("[ERROR]aclCreateDataBuffer failed");
                return 1;
            }

            ret = aclmdlAddDatasetBuffer(input, inputData);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]ACL_ModelInputDataAdd failed, ret[%d]", ret);
                return ret;
            }
        }

        aclmdlDataset *output = aclmdlCreateDataset();

        for (size_t outputIndex = 0; outputIndex < outputNum; outputIndex++) {
            size_t outSize = aclmdlGetOutputSizeByIndex(modelDesc_, outputIndex);
            void *outData;
            ret = aclrtMalloc(&outData, outSize, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Malloc device failed, ret[%d]", ret);
                return ret;
            }

            aclDataBuffer *outDataBuff = aclCreateDataBuffer((void *)outData, outSize);
            if (outDataBuff == nullptr) {
                LOG_ERROR("[ERROR]aclCreateDataBuffer failed");
                return 1;
            }

            ret = aclmdlAddDatasetBuffer(output, outDataBuff);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("ACL_ModelInputDataAdd failed, ret[%d]", ret);
                return ret;
            }
        }

        InferenceMem memInfo;
        memInfo.input = input;
        memInfo.output = output;
        memInfo.inferFile_vec = new std::vector<std::string>();

        if (cfg_->isDynamicAipp == true) {
            aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(maxBatch_);
            if (aippDynamicSet == nullptr) {
                LOG_ERROR("aclmdlCreateAIPP failed");
                return 1;
            }
            memInfo.dyAippSet = aippDynamicSet;
        } else {
            memInfo.dyAippSet = nullptr;
        }

        this->inferMemPool_vec_.push_back(memInfo);
    }

    LOG_INFO("Create memPool success, the size is %d", this->inferMemPool_vec_.size());

    return ACL_ERROR_NONE;
}


void Asyn_InferEngine::DestroyInferenceMemPool()
{
    aclError ret;

    ret = aclrtSetCurrentContext(this->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return;
    }

    for (size_t i = 0; i < inferMemPool_vec_.size(); i++) {
        aclmdlDataset *input = inferMemPool_vec_[i].input;
        DestroyDataset(input);

        aclmdlDataset *output = inferMemPool_vec_[i].output;
        DestroyDataset(output);

        delete inferMemPool_vec_[i].inferFile_vec;
    }

    LOG_INFO("DestroyInferenceMemPool success");
}

aclError Asyn_InferEngine::CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t imgInfoInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 1);
    uint32_t eachSize = sizeof(float) * 4;

    LOG_INFO("[INFO]imgInfoInputSize [%u] eachSize[%u]", imgInfoInputSize, eachSize);

    void *yoloImgInfo = nullptr;
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input, 1);
    yoloImgInfo = aclGetDataBufferAddr(dataBuffer);

    float imgInfo[4] = {0};
    imgInfo[0] = yoloImgInfo_.resizedWidth;
    imgInfo[1] = yoloImgInfo_.resizedHeight;

    int pos = 0;
    int errFlag = 0;

    for (int i = 0; i < fileName_vec->size(); i++) {
        std::string framename = fileName_vec->at(i);
        std::size_t dex = (framename).find_last_of(".");
        std::string inputFileName = (framename).erase(dex);
        std::size_t preci_dex = (inputFileName).find_first_of("_");
        if (std::string::npos != preci_dex) {
            inputFileName = (framename).erase(preci_dex);
        }

        std::string imgInfoMap_key = inputFileName + ".bin";

        std::unordered_map<std::string, std::pair<float, float>>::iterator iter =
            yoloImgInfo_.imgSizes_map.find(imgInfoMap_key);
        if (iter == yoloImgInfo_.imgSizes_map.end()) {
            LOG_ERROR("[INFO]not found key[%s] in map g_yolov3_imgInfos.imgSizes_map", imgInfoMap_key.c_str());
            return 1;
        }

        imgInfo[2] = (iter->second).first;
        imgInfo[3] = (iter->second).second;

        if (runMode_ == ACL_HOST) {
            ret = aclrtMemcpy((uint8_t *)yoloImgInfo + pos, eachSize, imgInfo, eachSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else {
            ret = aclrtMemcpy((uint8_t *)yoloImgInfo + pos, eachSize, imgInfo, eachSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        }

        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtMemcpy file [%s] imgInfo fail, ret[%d]", fileName_vec->at(i).c_str(), ret);
            return 1;
        }

        pos += eachSize;
    }

    return ACL_ERROR_NONE;
}


aclError Asyn_InferEngine::CreateInferInput(std::vector<std::string> &inferFile_vec)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t pos = 0;
    uint32_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    uint32_t singleImgSize = modelInputSize / maxBatch_;

    aclDataBuffer *dataBuffer =
        aclmdlGetDatasetBuffer(inferMemPool_vec_[memCurrent_ % cfg_->asynInferCfg.mem_pool_size].input, 0);
    void *p_batchDst = aclGetDataBufferAddr(dataBuffer);

    for (int i = 0; i < inferFile_vec.size(); i++) {
        printf("[INFO]img index %d, inputFolder %s, imgFile %s\n", i, cfg_->inputFolder.c_str(),
            inferFile_vec[i].c_str());
        std::string fileLocation = cfg_->inputFolder + "/" + inferFile_vec[i];
        FILE *pFile = fopen(fileLocation.c_str(), "r");
        if (pFile == nullptr) {
            LOG_ERROR("[ERROR]open file %s failed", fileLocation.c_str());
            continue;
        }
        printf("[INFO]load img index %d, img file name %s success\n", i, fileLocation.c_str());

        long fileSize = SdkInferGetFileSize(fileLocation.c_str());
        if (fileSize > singleImgSize || fileSize == 0) {
            LOG_ERROR("[ERROR]%s fileSize %ld * batch %d don't match with model inputSize %d or equal zero",
                fileLocation.c_str(), fileSize, cfg_->batchSize, modelInputSize / cfg_->batchSize);
            fclose(pFile);
            continue;
        }

        if (runMode_ == ACL_HOST) {
            void *buff = nullptr;
            ret = aclrtMallocHost(&buff, fileSize);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Malloc host buff failed[%d]", ret);
                fclose(pFile);
                continue;
            }

            fread(buff, sizeof(char), fileSize, pFile);
            fclose(pFile);

            ret = aclrtMemcpy((uint8_t *)p_batchDst + pos, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Copy host to device failed, ret[%d]", ret);
                aclrtFreeHost(buff);
                continue;
            }
            pos += fileSize;
            aclrtFreeHost(buff);
        } else {
            fread((uint8_t *)p_batchDst + pos, sizeof(char), fileSize, pFile);
            fclose(pFile);
            pos += fileSize;
        }
    }

    return ret;
}


aclError Asyn_InferEngine::ExecInference()
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t memPoolSize = cfg_->asynInferCfg.mem_pool_size;

    input_ = inferMemPool_vec_[memCurrent_ % memPoolSize].input;
    std::vector<std::string> *inferFile_vec = inferMemPool_vec_[memCurrent_ % memPoolSize].inferFile_vec;

    if (cfg_->modelType.compare("yolov3") == 0) {
        ret = CreateYoloImageInfoInput(input_, inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("YoloInferInputInit fail, ret[%d]", ret);
            return ret;
        }
    }

    if (cfg_->isDynamicBatch == true) {
        ret = SetDynamicBatch();
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("SetDynamicBatch fail, ret[%d]", ret);
            return ret;
        }
    }

    if (cfg_->isDynamicImg == true) {
        ret = SetDynamicImg();
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("SetDynamicImg fail, ret[%d]", ret);
            return ret;
        }
    }

    if (cfg_->isDynamicAipp == true) {
        ret = SetDynamicAipp();
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("SetDynamicImg fail, ret[%d]", ret);
            return ret;
        }
    }

    output_ = inferMemPool_vec_[memCurrent_ % memPoolSize].output;

    SdkInferGetTimeStart(&timeCost_, ASYN_MODEL_EXECUTE);
    ret = aclmdlExecuteAsync(modelId_, input_, output_, inferStream_);
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

        LOG_INFO("[INFO]Index %d call aclrtLaunchCallback sucess", memCurrent_ / cfg_->asynInferCfg.callback_interval);
    }

    return ret;
}


void Asyn_InferEngine::DumpTimeCost(std::ofstream &fstream)
{
    long long totalCost = 0;

    if (timeCost_.totalCount[ASYN_MODEL_EXECUTE] * cfg_->batchSize == 0) {
        LOG_ERROR("Exec inference count is 0, can not calculate performace statics");
        return;
    }

    for (int i = 0; i < POST_PROCESS; i++) {
        totalCost += timeCost_.totalTime[i];
    }

    totalCost -= timeCost_.totalTime[POST_PROCESS];

    long long avgCost = totalCost / (timeCost_.totalCount[ASYN_MODEL_EXECUTE] * cfg_->batchSize);
    LOG_INFO("%s totalCost %lld avgCost %lld", cfg_->modelType.c_str(), totalCost, avgCost);

    double avgMs = avgCost * 1.0 / 1000;
    char tmpCh[256];
    memset(tmpCh, 0, sizeof(tmpCh));
    snprintf(tmpCh, sizeof(tmpCh), "%s inference execute cost average time: %4.3f ms %4.3f fps/s\n",
        cfg_->modelType.c_str(), avgMs, (1000 / avgMs));

    LOG_INFO("%s", tmpCh);

    if (fstream.is_open()) {
        fstream << tmpCh;
    }
}

aclError Asyn_InferEngine::SetDynamicBatch()
{
    aclError ret = ACL_ERROR_NONE;
    size_t index;

    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &index);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlGetInputIndexByName failed, maybe static batch size, ret %d", ret);
        return ret;
    }

    LOG_INFO("#################################dynamic batch size index:%zd\n", index);

    ret = aclmdlSetDynamicBatchSize(modelId_, input_, index, cfg_->batchSize);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("dynamic batch set failed.");
    }

    return ret;
}

aclError Asyn_InferEngine::SetDynamicImg()
{
    aclError ret = ACL_ERROR_NONE;

    LOG_INFO("dynamic Img mode: width %ld, height %ld", cfg_->dynamicImgCfg.shapeW, cfg_->dynamicImgCfg.shapeH);

    size_t index;
    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &index);
    LOG_INFO("#################################dynamic img index:%zd", index);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlGetInputIndexByName failed, maybe static img shape, ret %d", ret);
        return ret;
    }

    ret = aclmdlSetDynamicHWSize(modelId_, input_, index, cfg_->dynamicImgCfg.shapeW, cfg_->dynamicImgCfg.shapeH);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("dynamic image shape set failed, ret %d", ret);
        return ret;
    }

    return ret;
}

aclError Asyn_InferEngine::SetDynamicAipp()
{
    aclError ret = ACL_ERROR_NONE;

    LOG_INFO("dynamic aipp mode.");
    size_t index;
    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_AIPP_NAME, &index);
    LOG_INFO("#################################dynamic aipp index:%zd", index);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlGetInputIndexByName failed, maybe static aipp, ret %d", ret);
        return ret;
    }

    uint32_t memPoolSize = cfg_->asynInferCfg.mem_pool_size;
    aippDynamicSet_ = inferMemPool_vec_[memCurrent_ % memPoolSize].dyAippSet;

    dynamic_aipp_config *dyAippCfg = &(cfg_->dynamicAippCfg);

    ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet_, dyAippCfg->srcImageSizeH, dyAippCfg->srcImageSizeW);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPSrcImageSize failed, w: %d, h: %d, ret: %d", dyAippCfg->srcImageSizeW,
            dyAippCfg->srcImageSizeH, ret);
        return ret;
    }
    ret = aclmdlSetAIPPInputFormat(aippDynamicSet_, dyAippCfg->inputFormat);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPInputFormat failed, ret %d", ret);
        return ret;
    }
    ret = aclmdlSetAIPPCscParams(aippDynamicSet_, dyAippCfg->cscParams.csc_switch, dyAippCfg->cscParams.cscMatrixR0C0,
        dyAippCfg->cscParams.cscMatrixR0C1, dyAippCfg->cscParams.cscMatrixR0C2, dyAippCfg->cscParams.cscMatrixR1C0,
        dyAippCfg->cscParams.cscMatrixR1C1, dyAippCfg->cscParams.cscMatrixR1C2, dyAippCfg->cscParams.cscMatrixR2C0,
        dyAippCfg->cscParams.cscMatrixR2C1, dyAippCfg->cscParams.cscMatrixR2C2, dyAippCfg->cscParams.cscOutputBiasR0,
        dyAippCfg->cscParams.cscOutputBiasR1, dyAippCfg->cscParams.cscOutputBiasR2, dyAippCfg->cscParams.cscInputBiasR0,
        dyAippCfg->cscParams.cscInputBiasR1, dyAippCfg->cscParams.cscInputBiasR2);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPCscParams failed, ret %d", ret);
        return ret;
    }
    ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet_, dyAippCfg->rbuvSwapSwitch);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPRbuvSwapSwitch failed, ret %d", ret);
        return ret;
    }
    ret = aclmdlSetAIPPAxSwapSwitch(aippDynamicSet_, dyAippCfg->axSwapSwitch);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPAxSwapSwitch failed, ret %d", ret);
        return ret;
    }

    for (size_t batchIndex = 0; batchIndex < maxBatch_; batchIndex++) {
        // config dtcPixelMean
        int dtcPixelMeanIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "dtcPixelMean");
        if (dtcPixelMeanIndex >= 0) {
            ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet_,
                dyAippCfg->dtcPixelMeanParams[dtcPixelMeanIndex].dtcPixelMeanChn0,
                dyAippCfg->dtcPixelMeanParams[dtcPixelMeanIndex].dtcPixelMeanChn1,
                dyAippCfg->dtcPixelMeanParams[dtcPixelMeanIndex].dtcPixelMeanChn2,
                dyAippCfg->dtcPixelMeanParams[dtcPixelMeanIndex].dtcPixelMeanChn3, batchIndex);
        } else {
            ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet_, 0, 0, 0, 0, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPDtcPixelMean failed, ret %d", ret);
            return ret;
        }

        // config dtcPixelMin
        int dtcPixelMinIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "dtcPixelMin");
        if (dtcPixelMinIndex >= 0) {
            ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet_,
                dyAippCfg->dtcPixelMinParams[dtcPixelMinIndex].dtcPixelMinChn0,
                dyAippCfg->dtcPixelMinParams[dtcPixelMinIndex].dtcPixelMinChn1,
                dyAippCfg->dtcPixelMinParams[dtcPixelMinIndex].dtcPixelMinChn2,
                dyAippCfg->dtcPixelMinParams[dtcPixelMinIndex].dtcPixelMinChn3, batchIndex);
        } else {
            ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet_, 0, 0, 0, 0, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPDtcPixelMin failed, ret %d", ret);
            return ret;
        }

        // config pixelVarReci
        int pixelVarReciIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "pixelVarReci");
        if (pixelVarReciIndex >= 0) {
            ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet_,
                dyAippCfg->pixelVarReciParams[pixelVarReciIndex].dtcPixelVarReciChn0,
                dyAippCfg->pixelVarReciParams[pixelVarReciIndex].dtcPixelVarReciChn1,
                dyAippCfg->pixelVarReciParams[pixelVarReciIndex].dtcPixelVarReciChn2,
                dyAippCfg->pixelVarReciParams[pixelVarReciIndex].dtcPixelVarReciChn3, batchIndex);
        } else {
            ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet_, 0.003921568627451, 0.003921568627451, 0.003921568627451,
                1.0, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPPixelVarReci failed, ret %d", ret);
            return ret;
        }

        // config crop
        int cropIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "crop");
        if (cropIndex >= 0) {
            ret = aclmdlSetAIPPCropParams(aippDynamicSet_, dyAippCfg->cropParams[cropIndex].cropSwitch,
                dyAippCfg->cropParams[cropIndex].cropStartPosW, dyAippCfg->cropParams[cropIndex].cropStartPosH,
                dyAippCfg->cropParams[cropIndex].cropSizeW, dyAippCfg->cropParams[cropIndex].cropSizeH, batchIndex);
        } else {
            ret = aclmdlSetAIPPCropParams(aippDynamicSet_, 0, 0, 0, 416, 416, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPCropParams failed, ret %d", ret);
            return ret;
        }

        // config padding
        int padIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "pad");
        if (padIndex >= 0) {
            ret = aclmdlSetAIPPPaddingParams(aippDynamicSet_, dyAippCfg->paddingParams[padIndex].paddingSwitch,
                dyAippCfg->paddingParams[padIndex].paddingSizeTop, dyAippCfg->paddingParams[padIndex].paddingSizeBottom,
                dyAippCfg->paddingParams[padIndex].paddingSizeLeft, dyAippCfg->paddingParams[padIndex].paddingSizeRight,
                batchIndex);
        } else {
            ret = aclmdlSetAIPPPaddingParams(aippDynamicSet_, 0, 0, 0, 0, 0, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPPaddingParams failed, ret %d", ret);
            return ret;
        }

        // config scf
        int scfIndex = GetDynamicAippParaByBatch(batchIndex, cfg_->dynamicAippCfg, "scf");
        if (scfIndex >= 0) {
            ret = aclmdlSetAIPPScfParams(aippDynamicSet_, dyAippCfg->scfParams[scfIndex].scfSwitch,
                dyAippCfg->scfParams[scfIndex].scfInputSizeW, dyAippCfg->scfParams[scfIndex].scfInputSizeH,
                dyAippCfg->scfParams[scfIndex].scfOutputSizeW, dyAippCfg->scfParams[scfIndex].scfOutputSizeH,
                batchIndex);
        } else {
            ret = aclmdlSetAIPPScfParams(aippDynamicSet_, 0, 1, 1, 1, 1, batchIndex);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclmdlSetAIPPPaddingParams failed, ret %d", ret);
            return ret;
        }
    }

    ret = aclmdlSetInputAIPP(modelId_, input_, index, aippDynamicSet_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetInputAIPP failed, ret %d", ret);
        return ret;
    }

    return ret;
}

/*
 * @brief thread entry of common asyn inference
 * @param [in] inferEngine: the pointer of asyn inference obj
 * @return none
 */
void AsynInferThread(Asyn_InferEngine *inferEngine)
{
    aclError ret = aclrtSetCurrentContext(inferEngine->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return;
    }
    LOG_INFO("[INFO]aclrtSetCurrentContext success");

    ret = aclrtCreateStream(&inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]create stream failed");
    }
    // create calback thread
    inferEngine->runFlag_ = 1;

    ret = pthread_create(&inferEngine->callbackPid_, nullptr, ThreadFunc, (void *)inferEngine);
    if (ret != 0) {
        LOG_ERROR("[ERROR]create thread failed, err = %d", ret);
    }
    LOG_INFO("[INFO]pthread_create  success");

    // Specifies the thread that processes the callback function on the stream
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
        for (int i = 0; i < inferEngine->files_.size(); i++) {
            if (cnt % batchSize == 0) {
                inferFile_vec = inferEngine->inferMemPool_vec_[inferEngine->memCurrent_ % memPoolSize].inferFile_vec;
                inferFile_vec->clear();
            }

            cnt++;
            inferFile_vec->push_back(inferEngine->files_[i]);

            if (cnt % batchSize == 0) {
                inferEngine->CreateInferInput(*inferFile_vec);

                LOG_INFO("[INFO]Index %d inference execute!", inferEngine->memCurrent_);

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

        loopCnt++;
    }
    // 残余数据一次送下去
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
        callbackData->inferEng = inferEngine;

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

    // unsubscribe report
    ret = aclrtUnSubscribeReport(static_cast<uint64_t>(inferEngine->callbackPid_), inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]acl unsubscribe report failed");
    }

    inferEngine->runFlag_ = 0;
    (void)pthread_join(static_cast<uint64_t>(inferEngine->callbackPid_), nullptr);

    ret = aclrtDestroyStream(inferEngine->inferStream_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtDestroyStream failed");
    }
}


Asyn_InferEngine::~Asyn_InferEngine()
{
    LOG_INFO("Asyn_InferEngine deconstruct.");
}

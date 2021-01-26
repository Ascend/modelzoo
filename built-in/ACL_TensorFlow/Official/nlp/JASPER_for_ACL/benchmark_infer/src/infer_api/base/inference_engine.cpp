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

aclError GetInferEngineConfig(Config *cfg, uint32_t chnIndex, std::string &modelType, std::string &inputFolder,
    std::string &outputFolder, std::string &om, aclrtContext &context, inferenceJsonConfig &jsonCfg)
{
    aclrtRunMode runMode = ACL_HOST;
    if (ACL_ERROR_NONE != aclrtGetRunMode(&runMode)) {
        LOG_ERROR("aclrtGetRunMode fail");
        return 1;
    }

    aclrtMemcpyKind cpyKind = ACL_MEMCPY_HOST_TO_HOST;
    if (ACL_DEVICE == runMode) {
        cpyKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }

    if (jsonCfg.commCfg.inferFlag != 1) {
        LOG_ERROR("inferFlag in json not set 1");
        return 1;
    }

    cfg->curChnNum = jsonCfg.inferCfg.channelNum;

    if (jsonCfg.commCfg.vpcFlag == 1) {
        cfg->preChnNum = jsonCfg.vpcCfg.vpc_channel_num;
    } else if (jsonCfg.commCfg.vdecFlag == 1) {
        cfg->preChnNum = jsonCfg.vdecCfgPara.channelNum;
    } else {
        cfg->preChnNum = 0;
    }

    cfg->context = context;
    cfg->om = om;
    cfg->inputFolder = inputFolder;
    cfg->resultFolder = outputFolder;
    cfg->modelType = modelType;
    cfg->inputArray = inference_json_cfg_tbl.dataCfg.dir_path_vec;

    cfg->loopNum = jsonCfg.commCfg.loopNum;
    cfg->frameWork = jsonCfg.commCfg.frame_work;

    cfg->postType = static_cast<POST_PROC_TYPE>(jsonCfg.inferCfg.postType);
    cfg->batchSize = jsonCfg.inferCfg.batch_size;
    cfg->inferType = static_cast<INFERENCE_TYPE>(jsonCfg.inferCfg.infer_type);
    cfg->imgType = jsonCfg.inferCfg.imgType;

    cfg->isDynamicBatch = ((jsonCfg.inferCfg.dynamicBathFlag == 1) ? true : false);
    cfg->isDynamicImg = ((jsonCfg.inferCfg.dynamicImgFlag == 1) ? true : false);
    cfg->isDynamicAipp = ((jsonCfg.inferCfg.dynamicAippFlag == 1) ? true : false);
    cfg->isDynamicDims = ((jsonCfg.inferCfg.dynamicDimsFlag == 1) ? true : false);

    if (cfg->modelType.compare(0, 6, "resnet") == 0 && cfg->postType == post_type_calc_accuracy) {
        cfg->resnetStdFile = jsonCfg.inferCfg.resnetStdFile;
    } else if (cfg->modelType.compare(0, 4, "yolo") == 0) {
        cfg->yoloImgInfoFile = jsonCfg.inferCfg.yoloImgInfoFile;
    }

    if (cfg->inferType != infer_type_syn) {
        cfg->asynInferCfg.mem_pool_size = jsonCfg.inferCfg.inferAsynPara.mem_pool_size;
        cfg->asynInferCfg.callback_interval = jsonCfg.inferCfg.inferAsynPara.callback_interval;
    }

    if (cfg->isDynamicImg == true) {
        cfg->dynamicImgCfg.shapeW = jsonCfg.inferCfg.dynamicImg.shapeW;
        cfg->dynamicImgCfg.shapeH = jsonCfg.inferCfg.dynamicImg.shapeH;
    }

    if (cfg->isDynamicDims == true) {
        cfg->dynamicDimsCfg.dydims.dimCount = jsonCfg.inferCfg.dynamicDims.dydims.dimCount;
        printf("dynamic dimCount is %d\n", cfg->dynamicDimsCfg.dydims.dimCount);
        for (int i = 0; i < cfg->dynamicDimsCfg.dydims.dimCount; i++) {
            cfg->dynamicDimsCfg.dydims.dims[i] = jsonCfg.inferCfg.dynamicDims.dydims.dims[i];
            printf("dims[%d] = %d\n", i, cfg->dynamicDimsCfg.dydims.dims[i]);
        }
    }

    if (cfg->isDynamicAipp == true) {
        aclrtMemcpy(&(cfg->dynamicAippCfg), sizeof(dynamic_aipp_config), &(jsonCfg.inferCfg.dynamicAippCfg),
            sizeof(dynamic_aipp_config), cpyKind);

        if (cfg->dynamicAippCfg.scfCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.scfParams, sizeof(aippScfConfig) * cfg->dynamicAippCfg.scfCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.scfParams, sizeof(aippScfConfig) * cfg->dynamicAippCfg.scfCfgNum,
                cpyKind);
        }

        if (cfg->dynamicAippCfg.cropCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.cropParams, sizeof(aippCropConfig) * cfg->dynamicAippCfg.cropCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.cropParams, sizeof(aippCropConfig) * cfg->dynamicAippCfg.cropCfgNum,
                cpyKind);
        }

        if (cfg->dynamicAippCfg.padCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.paddingParams, sizeof(aippPaddingConfig) * cfg->dynamicAippCfg.padCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.paddingParams,
                sizeof(aippPaddingConfig) * cfg->dynamicAippCfg.padCfgNum, cpyKind);
        }

        if (cfg->dynamicAippCfg.dtcPixelMeanCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.dtcPixelMeanParams,
                sizeof(aippDtcPixelMeanConfig) * cfg->dynamicAippCfg.dtcPixelMeanCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.dtcPixelMeanParams,
                sizeof(aippDtcPixelMeanConfig) * cfg->dynamicAippCfg.dtcPixelMeanCfgNum, cpyKind);
        }

        if (cfg->dynamicAippCfg.dtcPixelMinCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.dtcPixelMinParams,
                sizeof(aippDtcPixelMinConfig) * cfg->dynamicAippCfg.dtcPixelMinCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.dtcPixelMinParams,
                sizeof(aippDtcPixelMinConfig) * cfg->dynamicAippCfg.dtcPixelMinCfgNum, cpyKind);
        }

        if (cfg->dynamicAippCfg.pixelVarReciCfgNum) {
            aclrtMemcpy(cfg->dynamicAippCfg.pixelVarReciParams,
                sizeof(aippPixelVarReciConfig) * cfg->dynamicAippCfg.pixelVarReciCfgNum,
                jsonCfg.inferCfg.dynamicAippCfg.pixelVarReciParams,
                sizeof(aippPixelVarReciConfig) * cfg->dynamicAippCfg.pixelVarReciCfgNum, cpyKind);
        }
    }

    return ACL_ERROR_NONE;
}

int GetDynamicAippParaByBatch(size_t batchIndex, dynamic_aipp_config &dyAippCfg, std::string cfgItem)
{
    int i = 0;
    if (cfgItem.compare("dtcPixelMean") == 0) {
        for (i = 0; i < dyAippCfg.dtcPixelMeanCfgNum; i++) {
            if (batchIndex == dyAippCfg.dtcPixelMeanParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.dtcPixelMeanCfgNum) {
            return i;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("scf") == 0) {
        for (i = 0; i < dyAippCfg.scfCfgNum; i++) {
            if (batchIndex == dyAippCfg.scfParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.scfCfgNum) {
            return i;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("crop") == 0) {
        for (i = 0; i < dyAippCfg.cropCfgNum; i++) {
            if (batchIndex == dyAippCfg.cropParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.cropCfgNum) {
            return i;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("pad") == 0) {
        for (i = 0; i < dyAippCfg.padCfgNum; i++) {
            if (batchIndex == dyAippCfg.paddingParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.padCfgNum) {
            return i;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("dtcPixelMin") == 0) {
        for (i = 0; i < dyAippCfg.dtcPixelMinCfgNum; i++) {
            if (batchIndex == dyAippCfg.dtcPixelMinParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.dtcPixelMinCfgNum) {
            return i;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("pixelVarReci") == 0) {
        for (i = 0; i < dyAippCfg.pixelVarReciCfgNum; i++) {
            if (batchIndex == dyAippCfg.pixelVarReciParams[i].batchIndex) {
                break;
            }
        }

        if (i < dyAippCfg.pixelVarReciCfgNum) {
            return i;
        } else {
            return -1;
        }
    }

    return -1;
}

InferEngine::InferEngine() : inputDatas_(nullptr), outputDatas_(nullptr)
{
    runMode_ = ACL_HOST;
    if (ACL_ERROR_NONE != aclrtGetRunMode(&runMode_)) {
        LOG_ERROR("aclrtGetRunMode fail");
    }
    LOG_INFO("run mode: %d", runMode_);

    aclError ret = aclrtMemset(&timeCost_, sizeof(timeCost_), 0, sizeof(timeCost_));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtMemset timeCost_ failed");
    }
}

InferEngine::InferEngine(Config *config) : inputDatas_(nullptr), outputDatas_(nullptr), cfg_(config)
{
    runMode_ = ACL_HOST;
    if (ACL_ERROR_NONE != aclrtGetRunMode(&runMode_)) {
        LOG_ERROR("aclrtGetRunMode fail");
    }
    LOG_INFO("run mode: %d", runMode_);

    aclError ret = aclrtMemset(&timeCost_, sizeof(timeCost_), 0, sizeof(timeCost_));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtMemset timeCost_ failed");
    }
}

InferEngine::InferEngine(BlockingQueue<std::shared_ptr<Trans_Buff_T>> *inQue,
    BlockingQueue<std::shared_ptr<Trans_Buff_T>> *outQue)
    : inputDatas_(inQue), outputDatas_(outQue)
{
    runMode_ = ACL_HOST;
    if (ACL_ERROR_NONE != aclrtGetRunMode(&runMode_)) {
        LOG_ERROR("aclrtGetRunMode fail");
    }
    LOG_INFO("run mode: %d", runMode_);

    aclError ret = aclrtMemset(&timeCost_, sizeof(timeCost_), 0, sizeof(timeCost_));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclrtMemset timeCost_ failed");
    }
}

void InferEngine::DestroyDataset(aclmdlDataset *dataset)
{
    aclError ret;

    if (dataset == nullptr) {
        LOG_ERROR("dataset == null");
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        if (dataBuffer == nullptr) {
            LOG_ERROR("dataBuffer == null");
            continue;
        }

        void *data = aclGetDataBufferAddr(dataBuffer);
        if (data != nullptr) {
            ret = aclrtFree(data);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("aclrtFree data failed");
            }
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtFree dataBuffer failed");
        }
    }

    ret = aclmdlDestroyDataset(dataset);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtFree dataset failed");
    }

    return;
}

aclError InferEngine::LoadModel()
{
    size_t memSize;
    size_t weightsize;
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

    ret = aclmdlQuerySizeFromMem(modelData_, modelSize, &memSize, &weightsize);
    if (ACL_ERROR_NONE != ret) {
        UnloadModel();
        LOG_ERROR("query memory size failed, ret %d", ret);
        return ret;
    }

    dev_ptr_ = nullptr;
    ret = aclrtMalloc(&dev_ptr_, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_ERROR_NONE != ret) {
        UnloadModel();
        LOG_ERROR("alloc dev_ptr_ failed, ret %d", ret);
        return ret;
    }

    weight_ptr_ = nullptr;
    ret = aclrtMalloc(&weight_ptr_, weightsize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_ERROR_NONE != ret) {
        UnloadModel();
        LOG_ERROR("alloc weight_ptr_ failed, ret %d", ret);
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

void InferEngine::UnloadModel()
{
    aclError ret;

    ret = aclrtSetCurrentContext(this->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Set context failed");
        return;
    }

    if (modelDesc_ != nullptr) {
        ret = aclmdlDestroyDesc(modelDesc_);
        if (ret != ACL_ERROR_NONE) {
            printf("aclmdlDestroyDesc  failed, ret[%d]", ret);
        }
    }

    if (modelId_  != 0) {
        ret = aclmdlUnload(modelId_);
        if (ret != ACL_ERROR_NONE) {
            printf("aclmdlUnload  failed, ret[%d]", ret);
        }
    }

    if (modelData_ != nullptr) {
        delete[] modelData_;
    }

    if (dev_ptr_ != nullptr) {
        aclrtFree(dev_ptr_);
    }

    if (weight_ptr_ != nullptr) {
        aclrtFree(weight_ptr_);
    }

    printf("unload model success\n");
}

aclError InferEngine::Init(Config *config)
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
    LOG_INFO("[INFO] file num : %d ", fileNum);

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

    return 0;
}

aclError InferEngine::InitYolov3ImgInfo(std::string &yolo_ImgInfo_file)
{
    if (cfg_->modelType.compare(0, 4, "yolo") != 0) {
        return ACL_ERROR_NONE;
    }

    std::ifstream in(yolo_ImgInfo_file);
    if (!in) {
        printf("open yoloV3 img info file [%s] failed", yolo_ImgInfo_file.c_str());
        return 1;
    }

    std::string filename;
    float W, H;
    in >> W >> H;
    yoloImgInfo_.resizedWidth = W;
    yoloImgInfo_.resizedHeight = H;

    printf("yolov3 resized info: resizedWidth %d resizedHeight %d\n", yoloImgInfo_.resizedWidth,
        yoloImgInfo_.resizedHeight);

    while (in >> filename) {
        in >> W >> H;
        yoloImgInfo_.imgSizes_map[filename] = std::move(std::make_pair(W, H));
    }

    return ACL_ERROR_NONE;
}

aclError InferEngine::InitImgStdValue(std::string stdFilePath)
{
    if (cfg_->modelType.compare(0, 6, "resnet") != 0 || cfg_->postType != 1) {
        return ACL_ERROR_NONE;
    }

    aclError ret = aclrtSetCurrentContext(this->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return ret;
    }

    std::ifstream fin(stdFilePath);
    if (!fin) {
        LOG_ERROR("[ERROR]open resnet50StdFile failed, file name %s", stdFilePath.c_str());
        return -1;
    }

    std::string line_content;
    std::string fileName;
    int ctg;
    int cnt = 0;
    while (std::getline(fin, line_content)) {
        std::stringstream strStream(line_content);
        std::string tmp;
        int index = 0;
        while (std::getline(strStream, tmp, ',')) {
            if (index == 0) {
                fileName = tmp;
            } else {
                ctg = std::stoi(tmp);
                resnetTopRes_.cmp[fileName] = ctg;
            }
            index += 1;
        }

        if (cnt == 0) {
            std::size_t dex = (fileName).find_last_of(".");
            if (std::string::npos != dex) {
                resnetSubfix_ = fileName.substr(dex + 1);
            }

            printf("restnet standed file subfix: %s\n", resnetSubfix_.c_str());
        }

        cnt++;
    }

    resnetTopRes_.top1 = 0;
    resnetTopRes_.top5 = 0;
    resnetTopRes_.total = 0;

    fin.close();
    return 0;
}

aclError InferEngine::CreateYoloImageInfoInput(aclmdlDataset *input, std::vector<std::string> *fileName_vec)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t imgInfoInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 1);
    uint32_t eachSize = sizeof(float) * 4;

    LOG_INFO("[INFO]imgInfoInputSize [%u] eachSize[%u]", imgInfoInputSize, eachSize);

    void *yoloImgInfo = nullptr;
    ret = aclrtMalloc(&yoloImgInfo, imgInfoInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("malloc yolov3 imgInfo buf fail, ret[%d]", ret);
        return ret;
    }

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
            aclrtFree(yoloImgInfo);
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
            aclrtFree(yoloImgInfo);
            return ret;
        }

        pos += eachSize;
    }

    aclDataBuffer *imgInfoData = aclCreateDataBuffer((void *)yoloImgInfo, imgInfoInputSize);
    if (imgInfoData == nullptr) {
        LOG_ERROR("aclCreateDataBuffer failed");
        aclrtFree(yoloImgInfo);
        return 1;
    }

    ret = aclmdlAddDatasetBuffer(input, imgInfoData);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlAddDatasetBuffer failed, ret[%d]", ret);
        aclrtFree(yoloImgInfo);
        aclDestroyDataBuffer(imgInfoData);
        return ret;
    }

    return ACL_ERROR_NONE;
}

aclError InferEngine::CreateInferInput(std::vector<std::string> &inferFile_vec)
{
    aclError ret = ACL_ERROR_NONE;
    uint32_t pos = 0;

    void *dst;

    uint32_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    uint32_t singleImgSize = modelInputSize / maxBatch_;

    ret = aclrtMalloc(&dst, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Malloc device failed, ret[%d]", ret);
        return ret;
    }

    for (int i = 0; i < inferFile_vec.size(); i++) {
        std::string fileLocation = cfg_->inputFolder + "/" + inferFile_vec[i];
        FILE *pFile = fopen(fileLocation.c_str(), "r");
        if (pFile == nullptr) {
            LOG_ERROR("[ERROR]open file %s failed", fileLocation.c_str());
            continue;
        }

        LOG_INFO("[INFO]load img index %d, img file name %s success", i, fileLocation.c_str());
        long fileSize = SdkInferGetFileSize(fileLocation.c_str());

        if (fileSize > singleImgSize || fileSize == 0) {
            LOG_ERROR("[ERROR]%s fileSize %ld can not more than model each batch inputSize %d or equal zero",
                fileLocation.c_str(), fileSize, singleImgSize);
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

            ret = aclrtMemcpy((uint8_t *)dst + pos, fileSize, buff, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_ERROR_NONE) {
                LOG_ERROR("[ERROR]Copy host to device failed, ret[%d]", ret);
                aclrtFreeHost(buff);
                continue;
            }
            pos += fileSize;
            aclrtFreeHost(buff);
        } else {
            fread((uint8_t *)dst + pos, sizeof(char), fileSize, pFile);
            fclose(pFile);
            pos += fileSize;
        }
    }

    aclDataBuffer *inputData = aclCreateDataBuffer((void *)dst, modelInputSize);
    if (inputData == nullptr) {
        LOG_ERROR("aclCreateDataBuffer failed");
        aclrtFree(dst);
        return 1;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        LOG_ERROR("aclmdlCreateDataset failed, ret[%d]", ret);
        aclrtFree(dst);
        aclDestroyDataBuffer(inputData);
        return 1;
    }

    ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlAddDatasetBuffer failed, ret[%d]", ret);
        aclrtFree(dst);
        aclDestroyDataBuffer(inputData);
        aclmdlDestroyDataset(input_);
        return ret;
    }

    if (cfg_->modelType.compare("yolov3") == 0) {
        ret = CreateYoloImageInfoInput(input_, &inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("YoloInferInputInit fail, ret[%d]", ret);

            return ret;
        }
    }

    return ret;
}

aclError InferEngine::CreateInferOutput()
{
    aclError ret = ACL_ERROR_NONE;
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        LOG_ERROR("Create Output Dataset failed");
        return 1;
    }

    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    size_t i = 0;
    for (i = 0; i < outputNum; ++i) {
        uint64_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, (size_t)buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Malloc output host failed, ret[%d]", ret);
            break;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (outputData == nullptr) {
            LOG_ERROR("Create output data buffer failed");
            aclrtFree(outputBuffer);
            break;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Add output model dataset failed, ret[%d]", ret);
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            break;
        }
    }

    if (i < outputNum) {
        DestroyDataset(output_);
    }

    return ret;
}


aclError InferEngine::ExecInference()
{
    aclError ret = ACL_ERROR_NONE;

    if (cfg_->isDynamicBatch == true) {
        ret = SetDynamicBatch();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]SetDynamicBatch failed, ret %d", ret);
            return ret;
        }
    }

    if (cfg_->isDynamicImg == true) {
        ret = SetDynamicImg();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]SetDynamicImg failed, ret %d", ret);
            return ret;
        }
    }

    if (cfg_->isDynamicAipp == true) {
        ret = SetDynamicAipp();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]SetDynamicAipp failed, ret %d", ret);
            return ret;
        }
    }

    SdkInferGetTimeStart(&timeCost_, ASYN_MODEL_EXECUTE);
    ret = aclmdlExecute(modelId_, input_, output_);
    SdkInferGetTimeEnd(&timeCost_, ASYN_MODEL_EXECUTE);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]aclmdlExecute failed, ret %d", ret);
        if (cfg_->isDynamicAipp == true) {
            aclmdlDestroyAIPP(aippDynamicSet_);
        }
        return ret;
    }

    return ret;
}

int InferEngine::SaveInferResult(aclmdlDataset *output, std::vector<std::string> *inferFile_vec)
{
    aclError ret;

    aclrtSetCurrentContext(cfg_->context);

    std::string retFolder = cfg_->resultFolder + "/" + cfg_->modelType;

    DIR *op = opendir(retFolder.c_str());
    if (NULL == op) {
        mkdir(retFolder.c_str(), 00775);
    } else {
        closedir(op);
    }

    uint32_t batchSize = maxBatch_;
    void *outHostData = NULL;
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
#ifdef VERSION_C75_NOT_C73
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
#else
        uint32_t len = aclGetDataBufferSize(dataBuffer);
#endif

        if (runMode_ == ACL_HOST) {
            ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_ERROR_NONE) {
                printf("Malloc host failed.\n");
                return 1;
            }

            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_ERROR_NONE) {
                printf("Copy device to host failed.\n");
                aclrtFreeHost(outHostData);
                return 1;
            }
        }

        uint32_t eachSize = len / batchSize;
        for (size_t j = 0; j < inferFile_vec->size(); j++) {
            std::string framename = inferFile_vec->at(j);
            std::size_t dex = (framename).find_last_of(".");
            std::string inputFileName;
            if (std::string::npos != dex) {
                inputFileName = (framename).erase(dex);
            } else {
                inputFileName = framename;
            }

            std::string outFile;
            if (cfg_->modelType.compare(0, 5, "yolov") == 0) {
                if (cfg_->imgType.compare("rgb") == 0) {
                    std::size_t preci_dex = (inputFileName).find_first_of("_");
                    if (std::string::npos != preci_dex) {
                        inputFileName = (inputFileName).erase(preci_dex);
                    }
                }
                outFile = retFolder + "/" + "davinci_" + inputFileName + "_" + "output" + std::to_string(i) + ".bin";
            } else if (cfg_->modelType.compare(0, 6, "resnet") == 0) {
                if (cfg_->postType == 1) {
                    outFile = retFolder + "/" + inputFileName + "." + resnetSubfix_;
                } else {
                    outFile = retFolder + "/davinci_" + inputFileName + "_output.bin";
                }
            } else {
                outFile = retFolder + "/davinci_" + inputFileName + "_output0.bin";
            }

            printf("input file: [%s]save out file [%s]\n", inputFileName.c_str(), outFile.c_str());

            FILE *outputFile = fopen(outFile.c_str(), "wb");
            if (NULL == outputFile) {
                if (runMode_ == ACL_HOST) {
                    aclrtFreeHost(outHostData);
                }
                printf(" open file %s failed!\n", outFile.c_str());
                return 1;
            }

            if (runMode_ == ACL_HOST) {
                fwrite((uint8_t *)outHostData + (j * eachSize), eachSize, sizeof(char), outputFile);
            } else {
                fwrite((uint8_t *)data + (j * eachSize), eachSize, sizeof(char), outputFile);
            }

            fclose(outputFile);
        }

        if (runMode_ == ACL_HOST) {
            aclrtFreeHost(outHostData);
        }
    }

    return 0;
}

aclError InferEngine::GetModelInputOutputInfo()
{
    aclError ret;

    size_t inputNum = aclmdlGetNumInputs(modelDesc_);
    LOG_INFO("model input num %zd", inputNum);

    mdInputNum_ = inputNum;
    for (size_t i = 0; i < inputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++) {
        size_t size = aclmdlGetInputSizeByIndex(modelDesc_, i);
        mdInputInfo_[i].size = size;
        LOG_INFO("model input[%zd] size %zd", i, mdInputInfo_[i].size);

        aclmdlIODims dims;
        ret = aclmdlGetInputDims(modelDesc_, i, &dims);
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclmdlGetInputDims fail ret %d", ret);
            return 1;
        }

        mdInputInfo_[i].dimCount = dims.dimCount;
        if (runMode_ == ACL_HOST) {
            ret = aclrtMemcpy(mdInputInfo_[i].dims, mdInputInfo_[i].dimCount * sizeof(int64_t), dims.dims,
                mdInputInfo_[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        } else {
            ret = aclrtMemcpy(mdInputInfo_[i].dims, mdInputInfo_[i].dimCount * sizeof(int64_t), dims.dims,
                mdInputInfo_[i].dimCount * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclrtMemcpy fail ret %d line %d\n", ret, __LINE__);
            return 1;
        }

        LOG_INFO("model input[%zd] dimCount %zd", i, mdInputInfo_[i].dimCount);
        for (size_t dimIdx = 0; dimIdx < mdInputInfo_[i].dimCount; dimIdx++) {
            LOG_INFO("model input[%zd] dim[%zd] info %ld", i, dimIdx, mdInputInfo_[i].dims[dimIdx]);
        }

        mdInputInfo_[i].Format = aclmdlGetInputFormat(modelDesc_, i);

        mdInputInfo_[i].Type = aclmdlGetInputDataType(modelDesc_, i);

        LOG_INFO("model input[%zd] format %d inputType %d", i, mdInputInfo_[i].Format, mdInputInfo_[i].Type);

        mdInputInfo_[i].Name = aclmdlGetInputNameByIndex(modelDesc_, i);
        LOG_INFO("model input[%zd] name %s", i, mdInputInfo_[i].Name);

        size_t index;
        ret = aclmdlGetInputIndexByName(modelDesc_, mdInputInfo_[i].Name, &index);
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclmdlGetInputIndexByName fail ret %d line %d", ret, __LINE__);
            return 1;
        }

        if (i != index) {
            LOG_ERROR("aclmdlGetInputNameByIndex not equal aclmdlGetInputIndexByName");
            return 1;
        } else {
            LOG_INFO("model input name %s is belone to input %zd", mdInputInfo_[i].Name, index);
        }
    }

    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    LOG_INFO("model output num %zd", outputNum);

    mdOutputNum_ = outputNum;
    for (size_t i = 0; i < outputNum && i < MODEL_INPUT_OUTPUT_NUM_MAX; i++) {
        size_t size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        mdOutputInfo_[i].size = size;
        LOG_INFO("model output[%zd] size %zd", i, mdOutputInfo_[i].size);

        aclmdlIODims dims;
        ret = aclmdlGetOutputDims(modelDesc_, i, &dims);
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclmdlGetOutputDims fail ret %d", ret);
            return 1;
        }

        mdOutputInfo_[i].dimCount = dims.dimCount;
        if (runMode_ == ACL_HOST) {
            ret = aclrtMemcpy(mdOutputInfo_[i].dims, mdOutputInfo_[i].dimCount * sizeof(int64_t), dims.dims,
                mdOutputInfo_[i].dimCount * sizeof(int64_t), ACL_MEMCPY_HOST_TO_HOST);
        } else {
            ret = aclrtMemcpy(mdOutputInfo_[i].dims, mdOutputInfo_[i].dimCount * sizeof(int64_t), dims.dims,
                mdOutputInfo_[i].dimCount * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        }

        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclrtMemcpy fail ret %d line %d", ret, __LINE__);
            return 1;
        }

        LOG_INFO("model output[%zd] dimCount %zd", i, mdOutputInfo_[i].dimCount);

        for (size_t dimIdx = 0; dimIdx < mdOutputInfo_[i].dimCount; dimIdx++) {
            LOG_INFO("model output[%zd] dim[%zd] info %ld", i, dimIdx, mdOutputInfo_[i].dims[dimIdx]);
        }

        mdOutputInfo_[i].Format = aclmdlGetOutputFormat(modelDesc_, i);
        mdOutputInfo_[i].Type = aclmdlGetOutputDataType(modelDesc_, i);
        LOG_INFO("model output[%zd] format %d outputType %d", i, mdOutputInfo_[i].Format, mdOutputInfo_[i].Type);

        mdOutputInfo_[i].Name = aclmdlGetOutputNameByIndex(modelDesc_, i);
        LOG_INFO("model output[%zd] name %s", i, mdOutputInfo_[i].Name);

        size_t index;
        ret = aclmdlGetOutputIndexByName(modelDesc_, mdOutputInfo_[i].Name, &index);
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclmdlGetOutputIndexByName fail ret %d line %d", ret, __LINE__);
            return 1;
        }

        if (i != index) {
            LOG_ERROR("aclmdlGetOutputNameByIndex not equal aclmdlGetOutputIndexByName");
            return 1;
        } else {
            LOG_INFO("model output name %s is belone to output %d", mdOutputInfo_[i].Name, index);
        }

        ret = aclmdlGetCurOutputDims(modelDesc_, i, &dims);
        if (ACL_ERROR_NONE != ret) {
            LOG_ERROR("aclmdlGetCurOutputDims fail ret %d", ret);
            return 1;
        }
        LOG_INFO("aclmdlGetCurOutputDims output[%zd] dimCount %zd", i, dims.dimCount);
        for (size_t dimIdx = 0; dimIdx < dims.dimCount; dimIdx++) {
            LOG_INFO("aclmdlGetCurOutputDims output[%zd] dim[%zd] info %ld", i, dimIdx, dims.dims[dimIdx]);
        }
    }

    aclmdlBatch batch;
    ret = aclmdlGetDynamicBatch(modelDesc_, &batch);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("aclmdlGetDynamicBatch fail ret %d line %d", ret, __LINE__);
        return 1;
    }
    LOG_INFO("Dynamic batchCount %zd", batch.batchCount);

    maxBatch_ = cfg_->batchSize;
    dyModelInfo_.batchCount = batch.batchCount;
    for (size_t i = 0; i < batch.batchCount; i++) {
        if (batch.batch[i] > maxBatch_) {
            maxBatch_ = batch.batch[i];
        }
        dyModelInfo_.batch[i] = batch.batch[i];

        LOG_INFO("model support batch %ld", batch.batch[i]);
    }

    LOG_INFO("dynamic batch max batch size: %d", maxBatch_);

    aclmdlHW dyhw;
    ret = aclmdlGetDynamicHW(modelDesc_, -1, &dyhw);
    if (ACL_ERROR_NONE != ret) {
        LOG_ERROR("aclmdlGetDynamicHW fail ret %d line %d", ret, __LINE__);
        return 1;
    }
    LOG_INFO("Dynamic hwCount %zd", dyhw.hwCount);

    dyModelInfo_.hwCount = dyhw.hwCount;

    for (size_t hwIdx = 0; hwIdx < dyhw.hwCount; hwIdx++) {
        dyModelInfo_.hw[hwIdx][0] = dyhw.hw[hwIdx][0];
        dyModelInfo_.hw[hwIdx][1] = dyhw.hw[hwIdx][1];
        LOG_INFO("Dynamic shape model hwIndex[%zd] w %lu h %lu", hwIdx, dyhw.hw[hwIdx][0], dyhw.hw[hwIdx][0]);
    }

    return ACL_ERROR_NONE;
}

aclError InferEngine::SetDynamicBatch()
{
    aclError ret = ACL_ERROR_NONE;
    size_t index;

    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &index);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlGetInputIndexByName failed, maybe static batch size, ret %d", ret);
        return ret;
    }

    LOG_INFO("#################################dynamic batch size index:%zd", index);

    size_t batch_buffer_size = aclmdlGetInputSizeByIndex(modelDesc_, index);
    void *inputBuffer = NULL;
    ret = aclrtMalloc(&inputBuffer, batch_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtMalloc failed, ret[%d]", ret);
        return ret;
    }
    aclDataBuffer *inputBatchData = aclCreateDataBuffer(inputBuffer, batch_buffer_size);
    if (inputBatchData == NULL) {
        LOG_ERROR("aclCreateDataBuffer failed");
        return ret;
    }

    ret = aclmdlAddDatasetBuffer(input_, inputBatchData);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("ACL_ModelInputDataAdd failed, ret[%d]", ret);
        return ret;
    }

    ret = aclmdlSetDynamicBatchSize(modelId_, input_, index, cfg_->batchSize);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("dynamic batch set failed.");
    }

    return ret;
}

aclError InferEngine::SetDynamicImg()
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

    size_t shape_buffer_size = aclmdlGetInputSizeByIndex(modelDesc_, index);
    void *inputShapeBuffer = NULL;
    ret = aclrtMalloc(&inputShapeBuffer, shape_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtMalloc dynamic shape buff failed, ret %d", ret);
        return ret;
    }

    aclDataBuffer *shapeBuffer = aclCreateDataBuffer(inputShapeBuffer, shape_buffer_size);
    if (shapeBuffer == NULL) {
        LOG_ERROR("aclCreateDataBuffer shapeBuffer failed");
        aclrtFree(inputShapeBuffer);
        return ret;
    }

    ret = aclmdlAddDatasetBuffer(input_, shapeBuffer);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlAddDatasetBuffer shapeBuffer failed, ret[%d]", ret);
        aclrtFree(inputShapeBuffer);
        aclDestroyDataBuffer(shapeBuffer);
        return ret;
    }

    ret = aclmdlSetDynamicHWSize(modelId_, input_, index, cfg_->dynamicImgCfg.shapeW, cfg_->dynamicImgCfg.shapeH);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("dynamic image shape set failed, ret %d", ret);
        aclrtFree(inputShapeBuffer);
        aclDestroyDataBuffer(shapeBuffer);
        return ret;
    }

    return ret;
}

aclError InferEngine::SetDynamicAipp()
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

    aippDynamicSet_ = aclmdlCreateAIPP(maxBatch_);
    if (aippDynamicSet_ == nullptr) {
        LOG_ERROR("aclmdlCreateAIPP failed");
        return 1;
    }

    dynamic_aipp_config *dyAippCfg = &(cfg_->dynamicAippCfg);

    ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet_, dyAippCfg->srcImageSizeW, dyAippCfg->srcImageSizeH);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPSrcImageSize failed, w: %d, h: %d, ret: %d", dyAippCfg->srcImageSizeW,
            dyAippCfg->srcImageSizeH, ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        return ret;
    }
    ret = aclmdlSetAIPPInputFormat(aippDynamicSet_, dyAippCfg->inputFormat);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPInputFormat failed, ret %d", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
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
        aclmdlDestroyAIPP(aippDynamicSet_);
        return ret;
    }
    ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet_, dyAippCfg->rbuvSwapSwitch);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPRbuvSwapSwitch failed, ret %d", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        return ret;
    }
    ret = aclmdlSetAIPPAxSwapSwitch(aippDynamicSet_, dyAippCfg->axSwapSwitch);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetAIPPAxSwapSwitch failed, ret %d", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        return ret;
    }

    for (size_t batchIndex = 0; batchIndex < maxBatch_; batchIndex++) {
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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }

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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }

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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }

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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }

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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }

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
            aclmdlDestroyAIPP(aippDynamicSet_);
            return ret;
        }
    }

    size_t aippSize = aclmdlGetInputSizeByIndex(modelDesc_, index);
    void *aippAddr = nullptr;
    ret = aclrtMalloc(&aippAddr, aippSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtMalloc failed, ret[%d]", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        return ret;
    }

    aclDataBuffer *aipp_data = aclCreateDataBuffer(aippAddr, aippSize);
    ret = aclmdlAddDatasetBuffer(input_, aipp_data);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("ACL_ModelOutputDataAdd failed, ret[%d]", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        aclrtFree(aippAddr);
        aclDestroyDataBuffer(aipp_data);
        return ret;
    }
    ret = aclmdlSetInputAIPP(modelId_, input_, index, aippDynamicSet_);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclmdlSetInputAIPP failed, ret %d", ret);
        aclmdlDestroyAIPP(aippDynamicSet_);
        aclrtFree(aippAddr);
        aclDestroyDataBuffer(aipp_data);
        return ret;
    }

    return ret;
}

void InferThread(InferEngine *inferEngine)
{
    aclError ret = aclrtSetCurrentContext(inferEngine->cfg_->context);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("[ERROR]Set infer context failed");
        return;
    }
    LOG_INFO("[INFO]aclrtSetCurrentContext success");

    int cnt = 0;
    int batchSize = inferEngine->cfg_->batchSize;
    uint32_t loopCnt = 0;
    std::vector<std::string> inferFile_vec;

    while (loopCnt < inferEngine->cfg_->loopNum) {
        for (int i = 0; i < inferEngine->files_.size(); i++) {
            if (cnt % batchSize == 0) {
                inferFile_vec.clear();
            }

            cnt++;
            inferFile_vec.push_back(inferEngine->files_[i]);

            if (cnt % batchSize == 0) {
                ret = inferEngine->CreateInferInput(inferFile_vec);
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]CreateInferInput failed, ret %d", ret);
                    continue;
                }

                ret = inferEngine->CreateInferOutput();
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]CreateInferOutput failed, ret %d", ret);
                    inferEngine->DestroyDataset(inferEngine->input_);
                    continue;
                }

                ret = inferEngine->ExecInference();
                if (ret != ACL_ERROR_NONE) {
                    LOG_ERROR("[ERROR]ExecInference failed");
                    inferEngine->DestroyDataset(inferEngine->input_);
                    inferEngine->DestroyDataset(inferEngine->output_);
                    continue;
                }

                if (ret == ACL_ERROR_NONE) {
                    int ret32 = inferEngine->SaveInferResult(inferEngine->output_, &inferFile_vec);
                    if (ret != 0) {
                        LOG_ERROR("[ERROR]SaveInferResult failed, ret %d", ret32);
                    }

                    if (inferEngine->cfg_->isDynamicAipp == true) {
                        aclmdlDestroyAIPP(inferEngine->aippDynamicSet_);
                    }
                    inferEngine->DestroyDataset(inferEngine->input_);
                    inferEngine->DestroyDataset(inferEngine->output_);
                }
            }
        }

        loopCnt++;
    }
    if (cnt % batchSize != 0) {
        ret = inferEngine->CreateInferInput(inferFile_vec);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]CreateInferInput failed, ret %d", ret);
            return;
        }

        ret = inferEngine->CreateInferOutput();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]CreateInferOutput failed, ret %d", ret);
            inferEngine->DestroyDataset(inferEngine->input_);
            return;
        }

        ret = inferEngine->ExecInference();
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]AsynInferenceExecute failed");
            inferEngine->DestroyDataset(inferEngine->input_);
            inferEngine->DestroyDataset(inferEngine->output_);
            return;
        }

        int ret32 = inferEngine->SaveInferResult(inferEngine->output_, &inferFile_vec);
        if (ret != 0) {
            LOG_ERROR("[ERROR]SaveInferResult failed, ret %d", ret32);
        }

        inferEngine->DestroyDataset(inferEngine->input_);
        inferEngine->DestroyDataset(inferEngine->output_);
    }
}

aclError InferEngine::InferenceThreadProc()
{
    std::thread inferThread(InferThread, this);
    threads_.push_back(std::move(inferThread));
    return ACL_ERROR_NONE;
}

void InferEngine::join()
{
    std::for_each(threads_.begin(), threads_.end(), std::mem_fn(&std::thread::join));
}

InferEngine::~InferEngine()
{
    LOG_INFO("InferEngine deconstruct.");
}

void InferEngine::DumpTimeCost(std::ofstream &fstream)
{
    long long totalCost = 0;

    if (timeCost_.totalCount[ASYN_MODEL_EXECUTE] * cfg_->batchSize == 0) {
        LOG_ERROR("Exec inference count is 0, can not calculate performace statics");
        return;
    }

    totalCost += timeCost_.totalTime[ASYN_MODEL_EXECUTE];

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

void InferEngine::CalcTop(std::ofstream &file_stream)
{
    if (cfg_->modelType.compare(0, 6, "resnet") != 0 || cfg_->postType != 1) {
        return;
    }
    aclError ret;
    std::vector<std::string> result_file;
    int file_num = 0;

    std::string result_file_path = cfg_->resultFolder + "/" + cfg_->modelType;
    file_num = SdkInferScanFiles(result_file, result_file_path);
    int cnt = 0;
    for (int i = 0; i < file_num; i++) {
        std::string file_name = result_file[i];
        FILE *pFile = fopen((result_file_path + "/" + file_name).c_str(), "r");
        if (NULL == pFile) {
            printf("open file %s failed\n", file_name.c_str());
            continue;
        }
        long fileSize = SdkInferGetFileSize((result_file_path + "/" + file_name).c_str());
        if (fileSize == 0) {
            printf("file[%s] size is zero\n", file_name.c_str());
            continue;
        }
        void *buf = NULL;

        if (runMode_ == ACL_HOST) {
            ret = aclrtMallocHost(&buf, fileSize);
        } else {
            ret = aclrtMalloc(&buf, fileSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("[ERROR]malloc buf failed, ret %d line %d", ret, __LINE__);
        }

        fread((uint8_t *)buf, fileSize, sizeof(char), pFile);
        fclose(pFile);

        float *outData = NULL;
        outData = reinterpret_cast<float *>(buf);
        size_t single_batch_size = fileSize / sizeof(float);

        std::vector<int> resultVector(5, 0);
        for (int j = 1; j < single_batch_size; j++) {
            for (int k = 0; k < 5; k++) {
                int curPos = resultVector[k];
                if (outData[j] > outData[curPos]) {
                    for (int l = 4; l > k; l--) {
                        resultVector[l] = resultVector[l - 1];
                    }
                    resultVector[k] = j;
                    break;
                }
            }
        }

        resnetTopRes_.total++;
        for (int m = 0; m < 5; m++) {
            if (resnetTopRes_.cmp[file_name] == resultVector[0]) {
                resnetTopRes_.top1++;
                resnetTopRes_.top5++;
                break;
            } else if (resnetTopRes_.cmp[file_name] == resultVector[m]) {
                resnetTopRes_.top5++;
                break;
            }
        }
        if (runMode_ == ACL_HOST) {
            aclrtFreeHost(buf);
        } else {
            aclrtFree(buf);
        }

        outData = NULL;
    }

    char tmpCh[256] = {0};
    memset(tmpCh, 0, sizeof(tmpCh));
    snprintf(tmpCh, sizeof(tmpCh), "%s top1: %.06f top5: %.06f\n", cfg_->modelType.c_str(),
        resnetTopRes_.top1 / float(resnetTopRes_.total), resnetTopRes_.top5 / float(resnetTopRes_.total));
    file_stream << tmpCh;

    printf("%s", tmpCh);
    printf("%s top1Cnt: %d top5Cnt: %d totalCnt: %d\n\n", cfg_->modelType.c_str(), resnetTopRes_.top1,
        resnetTopRes_.top5, resnetTopRes_.total);
}
